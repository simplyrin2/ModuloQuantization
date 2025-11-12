#!/usr/bin/env python3
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


# --------------------------
# Utilities & seed
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------
# STE rounding
# --------------------------
class RoundSTEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through for rounding
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTEFn.apply(x)


# --------------------------
# Simple symmetric scale quantizer (no modulo)
# --------------------------
class ScaleQuantizer(nn.Module):
    """
    Symmetric fake-quantization:
      - Scale S is learnable (via softplus to keep positive).
      - q = clip(round(x / S), -Q, Q), Q=2^(b-1)-1
      - x_hat = S * q
    Supports per-tensor or per-channel (specify dim in forward).
    """
    def __init__(
        self,
        num_channels: Optional[int] = None,
        init_scale: float = 1.0,
        learnable: bool = True,
        per_channel: bool = False,
        bits: int = 8,
        min_scale: float = 1e-6,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.per_channel = per_channel
        self.num_channels = num_channels if per_channel else 1
        self.bits = int(bits)
        self.min_scale = float(min_scale)
        self.use_softplus = bool(use_softplus)

        if per_channel and num_channels is None:
            raise ValueError("num_channels must be specified when per_channel=True")

        shape = (self.num_channels,) if per_channel else ()

        # Parameterization: softplus for positivity
        if self.use_softplus:
            # inverse softplus approx to init raw param
            def inv_softplus(y):
                return torch.log(torch.expm1(torch.tensor(y)))
            init_raw = float(inv_softplus(init_scale)) if init_scale > 1e-6 else -20.0
            self.scale_param = nn.Parameter(
                torch.full(shape, init_raw, dtype=torch.float32),
                requires_grad=learnable,
            )
        else:
            self.scale_param = nn.Parameter(
                torch.full(shape, float(init_scale), dtype=torch.float32),
                requires_grad=learnable,
            )

    @property
    def qmax(self) -> float:
        return float((1 << (self.bits - 1)) - 1)

    @property
    def scale(self) -> torch.Tensor:
        if self.use_softplus:
            # softplus + epsilon
            return F.softplus(self.scale_param) + self.min_scale
        else:
            return torch.clamp(self.scale_param, min=self.min_scale)

    def _broadcast(self, t: torch.Tensor, x: torch.Tensor, dim: Optional[int]):
        if not self.per_channel:
            return t
        if dim is None:
            raise ValueError("dim must be provided for per-channel mode")
        shape = [1] * x.ndim
        shape[dim] = self.num_channels
        return t.view(*shape)

    def forward(self, x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        s = self._broadcast(self.scale, x, dim)
        q = torch.clamp(round_ste(x / s), min=-self.qmax, max=self.qmax)
        x_hat = s * q
        return x_hat

    @torch.no_grad()
    def set_scale(self, new_scale: torch.Tensor):
        """
        Set scale value directly (for calibration).
        new_scale shape should match (num_channels,) or ().
        """
        new_scale = torch.clamp(new_scale, min=self.min_scale)
        if self.use_softplus:
            # find raw such that softplus(raw) ~= new_scale
            raw = torch.log(torch.expm1(new_scale - self.min_scale))
            self.scale_param.data.copy_(raw.to(self.scale_param.dtype))
        else:
            self.scale_param.data.copy_(new_scale.to(self.scale_param.dtype))


# --------------------------
# QuantLinear wrapper for simple scale quant
# --------------------------
class QuantLinearScale(nn.Module):
    """
    Linear layer with simple symmetric scale fake-quantization (no modulo).
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        quantize_input: bool = False,
        learnable_scale: bool = True,
        init_scale: float = 1.0,
        per_channel: bool = False,
        bits: int = 8,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias_flag = base_linear.bias is not None
        self.per_channel = per_channel

        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        if self.bias_flag:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.w_quant = ScaleQuantizer(
            num_channels=self.out_features if per_channel else None,
            init_scale=init_scale,
            learnable=learnable_scale,
            per_channel=per_channel,
            bits=bits,
        )
        self.a_quant = ScaleQuantizer(
            num_channels=self.in_features if per_channel else None,
            init_scale=init_scale,
            learnable=learnable_scale,
            per_channel=per_channel,
            bits=bits,
        ) if quantize_input else None

        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a_quant is not None:
            x_q = self.a_quant(x, dim=-1) if self.per_channel else self.a_quant(x)
        else:
            x_q = x
        w_q = self.w_quant(self.weight, dim=0) if self.per_channel else self.w_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def calibrate_scale_from_data(self, xs: List[torch.Tensor], percentile: float = 0.999):
        """
        Calibrate S using percentile(|x|)/Q. For weights, similar per-tensor/channel.
        """
        Q = self.w_quant.qmax

        # Activations
        if self.a_quant is not None and len(xs) > 0:
            if self.per_channel:
                cat = torch.cat([t.detach().reshape(-1, self.in_features) for t in xs], dim=0).abs()
                s_vals = torch.empty(self.in_features, dtype=torch.float32)
                for c in range(self.in_features):
                    vals = cat[:, c].reshape(-1)
                    k = max(1, int(math.ceil(percentile * vals.numel())))
                    topk, _ = torch.kthvalue(vals, k)
                    s_vals[c] = max(float(topk) / Q, 1e-3)
                self.a_quant.set_scale(s_vals)
            else:
                cat = torch.cat([t.detach().reshape(-1) for t in xs], dim=0).abs()
                k = max(1, int(math.ceil(percentile * cat.numel())))
                topk, _ = torch.kthvalue(cat, k)
                s = max(float(topk) / Q, 1e-3)
                self.a_quant.set_scale(torch.tensor(s))

        # Weights
        if self.per_channel:
            wabs = self.weight.detach().abs()  # [out_features, in_features]
            s_vals = torch.empty(self.out_features, dtype=torch.float32)
            for c in range(self.out_features):
                vals = wabs[c, :].reshape(-1)
                k2 = max(1, int(math.ceil(percentile * vals.numel())))
                topk2, _ = torch.kthvalue(vals, k2)
                s_vals[c] = max(float(topk2) / Q, 1e-3)
            self.w_quant.set_scale(s_vals)
        else:
            wabs = self.weight.detach().abs().reshape(-1)
            k2 = max(1, int(math.ceil(percentile * wabs.numel())))
            topk2, _ = torch.kthvalue(wabs, k2)
            s = max(float(topk2) / Q, 1e-3)
            self.w_quant.set_scale(torch.tensor(s))

        self.calibrated.data.fill_(1)


# --------------------------
# Replace Linear with QuantLinearScale in model
# --------------------------
def replace_linear_with_scalequantlinear(
    module: nn.Module,
    quantize_input: bool = False,
    learnable_scale: bool = True,
    init_scale: float = 1.0,
    per_channel: bool = False,
    bits: int = 8,
    skip_keys: Optional[List[str]] = None,
):
    if skip_keys is None:
        skip_keys = []

    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear) and all(kw not in full_name for kw in skip_keys):
            setattr(module, name, QuantLinearScale(
                child, quantize_input, learnable_scale,
                init_scale, per_channel, bits
            ))
        else:
            replace_linear_with_scalequantlinear(
                child, quantize_input, learnable_scale,
                init_scale, per_channel, bits, skip_keys
            )


# --------------------------
# Calibration pass
# --------------------------
@torch.no_grad()
def calibrate_model_scale(model: nn.Module, loader: DataLoader, device: torch.device,
                          num_batches: int = 20, percentile: float = 0.999):
    """
    Collect activations via hooks and calibrate scales.
    """
    model.eval()
    inp_buffers: Dict[nn.Module, List[torch.Tensor]] = {}

    def make_hook(mod):
        def hook(mod, inputs):
            x = inputs[0]
            if mod not in inp_buffers:
                inp_buffers[mod] = []
            xs = x[:2].detach().cpu() if x.dim() >= 2 else x.detach().cpu()
            inp_buffers[mod].append(xs)
        return hook

    handles = []
    for m in model.modules():
        if isinstance(m, QuantLinearScale):
            handles.append(m.register_forward_pre_hook(make_hook(m)))

    seen = 0
    for batch in loader:
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        seen += 1
        if seen >= num_batches:
            break

    for h in handles:
        h.remove()

    # Calibrate
    for m in model.modules():
        if isinstance(m, QuantLinearScale):
            xs = inp_buffers.get(m, [])
            m.calibrate_scale_from_data(xs, percentile=percentile)


# --------------------------
# Log scale parameters
# --------------------------
@torch.no_grad()
def log_scale_params(model: nn.Module, step: int, verbose: bool = True):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearScale):
            # weights
            s_w = module.w_quant.scale.data
            stats[f"{name}/w_scale_mean"] = s_w.mean().item()
            stats[f"{name}/w_scale_min"] = s_w.min().item()
            stats[f"{name}/w_scale_max"] = s_w.max().item()
            if s_w.numel() > 1:
                stats[f"{name}/w_scale_std"] = s_w.std().item()

            # activations
            if module.a_quant is not None:
                s_a = module.a_quant.scale.data
                stats[f"{name}/a_scale_mean"] = s_a.mean().item()
                stats[f"{name}/a_scale_min"] = s_a.min().item()
                stats[f"{name}/a_scale_max"] = s_a.max().item()
                if s_a.numel() > 1:
                    stats[f"{name}/a_scale_std"] = s_a.std().item()

            if verbose:
                print(f"\n{name}:")
                print(f"  Weight - S: mean={s_w.mean().item():.6g}, min={s_w.min().item():.6g}, max={s_w.max().item():.6g}")
                if module.a_quant is not None:
                    print(f"  Activ  - S: mean={s_a.mean().item():.6g}, min={s_a.min().item():.6g}, max={s_a.max().item():.6g}")

    # wandb.log(stats, step=step)
    return stats


# --------------------------
# Save model + scales
# --------------------------
def save_model(model: nn.Module, tokenizer, save_dir: str, epoch: int, ppl: float):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"model_epoch{epoch}_ppl{ppl:.2f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ppl': ppl,
    }, model_path)
    print(f"Model saved to {model_path}")

    tokenizer.save_pretrained(save_dir)

    # Save scales
    scale_params = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearScale):
            scale_params[f"{name}.w_quant.scale"] = module.w_quant.scale.detach().cpu()
            if module.a_quant is not None:
                scale_params[f"{name}.a_quant.scale"] = module.a_quant.scale.detach().cpu()
    spath = os.path.join(save_dir, f"scale_params_epoch{epoch}.pt")
    torch.save(scale_params, spath)
    print(f"Scale parameters saved to {spath}")


@torch.no_grad()
def collate_for_lm(batch, tokenizer, max_length=None):
    batch = [item for item in batch if len(item["input_ids"]) > 0]
    if len(batch) == 0:
        return None

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


# --------------------------
# Training config
# --------------------------
@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 512
    per_device_train_batch_size: int = 20
    per_device_eval_batch_size: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    seed: int = 42
    quantize_input: bool = False
    learnable_scale: bool = True
    init_scale: float = 0.05  # typical small initial scale; calibration adjusts it
    per_channel: bool = True
    calibrate_batches: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./saved_models_scale"
    log_scale_params_freq: int = 100
    bits: int = 8


# --------------------------
# Main
# --------------------------
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # dataset
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")

    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]

    def collate_fn(batch):
        return collate_for_lm(batch, tokenizer, max_length=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Replace Linear with QuantLinearScale
    quant_mode = "per-channel" if cfg.per_channel else "per-tensor"
    print(f"[Baseline] Simple scaling quantization ({quant_mode})")
    replace_linear_with_scalequantlinear(
        model,
        quantize_input=cfg.quantize_input,
        learnable_scale=cfg.learnable_scale,
        init_scale=cfg.init_scale,
        per_channel=cfg.per_channel,
        bits=cfg.bits,
        skip_keys=[],
    )
    model.to(device)

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)

    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
    num_warmup = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # calibration
    print("Calibrating scales from validation batches...")
    calibrate_model_scale(model, val_loader, device, num_batches=cfg.calibrate_batches)

    print("\nInitial scale parameters after calibration:")
    log_scale_params(model, step=0, verbose=True)

    model.train()
    print(f"Trainable parameters: {count_parameters(model)}")
    step = 0
    best_ppl = float("inf")

    wandb.init(project=f"scale-quantization-{quant_mode}", config=vars(cfg))
    wandb.watch(model, log='parameters', log_freq=100)

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / cfg.grad_accum_steps).backward()
            if (i + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # if step % cfg.log_scale_params_freq == 0:
                #     stats = log_scale_params(model, step=step, verbose=False)
                #     wandb.log(stats, step=step)

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "step": step})

        # eval
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                n_batches += 1

        if n_batches == 0:
            avg_val_loss = float("nan")
            ppl = float("nan")
        else:
            avg_val_loss = val_loss / n_batches
            ppl = math.exp(avg_val_loss)

        print(f"\nEpoch {epoch+1} | train_loss={epoch_loss/len(train_loader):.4f} | val_loss={avg_val_loss:.4f} | ppl={ppl:.4f}")

        print(f"\nScale parameters after epoch {epoch+1}:")
        stats = log_scale_params(model, step=step, verbose=True)
        wandb.log(stats)
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "ppl": ppl, "step": step})

        if ppl < best_ppl:
            best_ppl = ppl
            save_model(model, tokenizer, cfg.save_dir, epoch+1, ppl)

        save_model(model, tokenizer, os.path.join(cfg.save_dir, "checkpoints"), epoch+1, ppl)
        model.train()

    print(f"\nBest val PPL: {best_ppl}")
    wandb.log({"best_val_ppl": best_ppl})
    save_model(model, tokenizer, os.path.join(cfg.save_dir, "final"), cfg.num_epochs, ppl)
    wandb.finish()


if __name__ == "__main__":
    main()