# full_script_modulo_qat_gpt2_ppl.py
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Hugging Face
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
# Modulo STE (unchanged)
# --------------------------
class ModuloSTEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulus):
        m = torch.clamp(modulus, min=1e-8)
        y = torch.remainder(x, m)  # maps into [0, m)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def modulo_ste(x, modulus: torch.Tensor):
    return ModuloSTEFn.apply(x, modulus)

# --------------------------
# ModuloQuantizer - Supports both per-tensor and per-channel
# --------------------------
class ModuloQuantizer(nn.Module):
    """Modulo quantizer supporting both per-tensor and per-channel modes."""
    def __init__(self, num_channels: Optional[int] = None, init_lambda: float = 1.0, 
                 init_z: float = 8.0, learnable: bool = True, per_channel: bool = False):
        super().__init__()
        self.per_channel = per_channel
        self.num_channels = num_channels if per_channel else 1
        
        if per_channel and num_channels is None:
            raise ValueError("num_channels must be specified when per_channel=True")
        
        # Create parameters with appropriate shape
        param_shape = (self.num_channels,) if per_channel else ()
        
        self.lambda_param = nn.Parameter(
            torch.full(param_shape, float(init_lambda)), 
            requires_grad=learnable
        )
        self.z_param = nn.Parameter(
            torch.full(param_shape, float(init_z)), 
            requires_grad=learnable
        )

    @property
    def modulus(self):
        """Returns modulus values (scalar for per-tensor, vector for per-channel)."""
        return torch.clamp(self.z_param * self.lambda_param, min=1e-6)

    def forward(self, x, dim: Optional[int] = None):
        """
        Apply quantization.
        
        Args:
            x: Input tensor
            dim: Channel dimension (only used in per-channel mode)
                 For weights: dim=0 (output channels)
                 For activations: dim=-1 (last dimension)
        """
        modulus = self.modulus
        
        if self.per_channel:
            if dim is None:
                raise ValueError("dim must be specified in per-channel mode")
            # Reshape modulus to broadcast correctly along the channel dimension
            shape = [1] * x.ndim
            shape[dim] = self.num_channels
            modulus = modulus.view(*shape)
        
        return modulo_ste(x, modulus)

# --------------------------
# QuantLinear wrapper (weights/activations)
# --------------------------
class QuantLinear(nn.Module):
    """Linear layer with modulo quantization (per-tensor or per-channel)."""
    def __init__(self, base_linear: nn.Linear, quantize_input: bool = False, 
                 learnable_modulus: bool = True, init_lambda: float = 1.0, 
                 init_z: float = 8.0, per_channel: bool = False):
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

        # Weight quantizer (per-channel: one modulus per output channel, per-tensor: single modulus)
        self.w_quant = ModuloQuantizer(
            num_channels=self.out_features if per_channel else None,
            init_lambda=init_lambda, 
            init_z=init_z, 
            learnable=learnable_modulus,
            per_channel=per_channel
        )
        
        # Activation quantizer (per-channel: one modulus per input feature, per-tensor: single modulus)
        self.a_quant = ModuloQuantizer(
            num_channels=self.in_features if per_channel else None,
            init_lambda=init_lambda, 
            init_z=init_z, 
            learnable=learnable_modulus,
            per_channel=per_channel
        ) if quantize_input else None

        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Quantize input activations
        if self.a_quant is not None:
            x_q = self.a_quant(x, dim=-1) if self.per_channel else self.a_quant(x)
        else:
            x_q = x

        # Quantize weights
        w_q = self.w_quant(self.weight, dim=0) if self.per_channel else self.w_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def calibrate_modulus_from_data(self, xs: List[torch.Tensor], percentile: float = 0.999):
        """Calibrate modulus based on data statistics (per-tensor or per-channel)."""
        
        # Calibrate activation quantizer
        if self.a_quant is not None and len(xs) > 0:
            if self.per_channel:
                # Per-channel calibration for activations
                cat = torch.cat([t.detach().reshape(-1, self.in_features) for t in xs], dim=0)
                cat_abs = cat.abs()
                
                for c in range(self.in_features):
                    channel_vals = cat_abs[:, c]
                    k = max(1, int(math.ceil(percentile * channel_vals.numel())))
                    topk, _ = torch.kthvalue(channel_vals, k)
                    new_mod = max(2.0 * float(topk), 1e-3)
                    
                    self.a_quant.z_param.data[c] = 1.0
                    self.a_quant.lambda_param.data[c] = new_mod
            else:
                # Per-tensor calibration for activations
                cat = torch.cat([t.detach().reshape(-1) for t in xs], dim=0).abs()
                k = max(1, int(math.ceil(percentile * cat.numel())))
                topk, _ = torch.kthvalue(cat, k)
                new_mod = max(2.0 * float(topk), 1e-3)
                
                self.a_quant.z_param.data.fill_(1.0)
                self.a_quant.lambda_param.data.fill_(new_mod)

        # Calibrate weight quantizer
        if self.per_channel:
            # Per-channel calibration for weights
            wabs = self.weight.detach().abs()  # [out_features, in_features]
            
            for c in range(self.out_features):
                channel_weights = wabs[c, :]
                k2 = max(1, int(math.ceil(percentile * channel_weights.numel())))
                topk2, _ = torch.kthvalue(channel_weights.reshape(-1), k2)
                new_mod_w = max(2.0 * float(topk2), 1e-3)
                
                self.w_quant.z_param.data[c] = 1.0
                self.w_quant.lambda_param.data[c] = new_mod_w
        else:
            # Per-tensor calibration for weights
            wabs = self.weight.detach().abs().reshape(-1)
            k2 = max(1, int(math.ceil(percentile * wabs.numel())))
            topk2, _ = torch.kthvalue(wabs, k2)
            new_mod_w = max(2.0 * float(topk2), 1e-3)
            
            self.w_quant.z_param.data.fill_(1.0)
            self.w_quant.lambda_param.data.fill_(new_mod_w)

        self.calibrated.data.fill_(1)

# --------------------------
# Replace Linear with QuantLinear in model
# --------------------------
def replace_linear_with_quantlinear(module: nn.Module,
                                    quantize_input: bool = False,
                                    learnable_modulus: bool = True,
                                    init_lambda: float = 1.0,
                                    init_z: float = 8.0,
                                    per_channel: bool = False,
                                    skip_keys: Optional[List[str]] = None):
    if skip_keys is None:
        skip_keys = []

    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear) and all(kw not in full_name for kw in skip_keys):
            setattr(module, name, QuantLinear(
                child, quantize_input, learnable_modulus, 
                init_lambda, init_z, per_channel
            ))
        else:
            replace_linear_with_quantlinear(
                child, quantize_input, learnable_modulus, 
                init_lambda, init_z, per_channel, skip_keys
            )

# --------------------------
# Calibration: collect inputs to QuantLinear modules
# --------------------------
@torch.no_grad()
def calibrate_model_modulus(model: nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 50):
    model.eval()
    handles = []
    activations: Dict[int, List[torch.Tensor]] = {}

    def make_hook(mod):
        idx = id(mod)
        activations[idx] = []
        def hook(module, inp, out):
            x = inp[0].detach()
            if x.is_cuda:
                x = x.cpu()
            if len(activations[idx]) < 8:
                activations[idx].append(x)
        return hook

    for m in model.modules():
        if isinstance(m, QuantLinear):
            h = m.register_forward_hook(make_hook(m))
            handles.append(h)

    it = iter(dataloader)
    for step in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)

    for h in handles:
        h.remove()

    for m in model.modules():
        if isinstance(m, QuantLinear):
            xs = activations.get(id(m), [])
            m.calibrate_modulus_from_data(xs, percentile=0.999)

    model.train()

# --------------------------
# Log quantization parameters
# --------------------------
@torch.no_grad()
def log_quantization_params(model: nn.Module, step: int, verbose: bool = True):
    """Log quantization parameters (lambda and z) to wandb and optionally print them."""
    quant_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # Weight quantizer stats
            w_lambda = module.w_quant.lambda_param.data
            w_z = module.w_quant.z_param.data
            w_modulus = module.w_quant.modulus.data
            
            quant_stats[f"{name}/w_lambda_mean"] = w_lambda.mean().item()
            quant_stats[f"{name}/w_lambda_std"] = w_lambda.std().item() if w_lambda.numel() > 1 else 0.0
            quant_stats[f"{name}/w_lambda_min"] = w_lambda.min().item()
            quant_stats[f"{name}/w_lambda_max"] = w_lambda.max().item()
            
            quant_stats[f"{name}/w_z_mean"] = w_z.mean().item()
            quant_stats[f"{name}/w_z_std"] = w_z.std().item() if w_z.numel() > 1 else 0.0
            quant_stats[f"{name}/w_z_min"] = w_z.min().item()
            quant_stats[f"{name}/w_z_max"] = w_z.max().item()
            
            quant_stats[f"{name}/w_modulus_mean"] = w_modulus.mean().item()
            quant_stats[f"{name}/w_modulus_min"] = w_modulus.min().item()
            quant_stats[f"{name}/w_modulus_max"] = w_modulus.max().item()
            
            # Activation quantizer stats (if exists)
            if module.a_quant is not None:
                a_lambda = module.a_quant.lambda_param.data
                a_z = module.a_quant.z_param.data
                a_modulus = module.a_quant.modulus.data
                
                quant_stats[f"{name}/a_lambda_mean"] = a_lambda.mean().item()
                quant_stats[f"{name}/a_lambda_std"] = a_lambda.std().item() if a_lambda.numel() > 1 else 0.0
                quant_stats[f"{name}/a_lambda_min"] = a_lambda.min().item()
                quant_stats[f"{name}/a_lambda_max"] = a_lambda.max().item()
                
                quant_stats[f"{name}/a_z_mean"] = a_z.mean().item()
                quant_stats[f"{name}/a_z_std"] = a_z.std().item() if a_z.numel() > 1 else 0.0
                quant_stats[f"{name}/a_z_min"] = a_z.min().item()
                quant_stats[f"{name}/a_z_max"] = a_z.max().item()
                
                quant_stats[f"{name}/a_modulus_mean"] = a_modulus.mean().item()
                quant_stats[f"{name}/a_modulus_min"] = a_modulus.min().item()
                quant_stats[f"{name}/a_modulus_max"] = a_modulus.max().item()
            
            if verbose:
                print(f"\n{name}:")
                print(f"  Weight - lambda: {w_lambda.mean().item():.4f} ± {w_lambda.std().item():.4f}, "
                      f"z: {w_z.mean().item():.4f} ± {w_z.std().item():.4f}, "
                      f"modulus: {w_modulus.mean().item():.4f}")
                if module.a_quant is not None:
                    print(f"  Activation - lambda: {a_lambda.mean().item():.4f} ± {a_lambda.std().item():.4f}, "
                          f"z: {a_z.mean().item():.4f} ± {a_z.std().item():.4f}, "
                          f"modulus: {a_modulus.mean().item():.4f}")
    
    # Log to wandb
    wandb.log(quant_stats, step=step)
    return quant_stats

# --------------------------
# Save model
# --------------------------
def save_model(model: nn.Module, tokenizer, save_dir: str, epoch: int, ppl: float):
    """Save model, tokenizer, and quantization parameters."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, f"model_epoch{epoch}_ppl{ppl:.2f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ppl': ppl,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save quantization parameters separately
    quant_params = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            quant_params[f"{name}.w_quant.lambda_param"] = module.w_quant.lambda_param.data.cpu()
            quant_params[f"{name}.w_quant.z_param"] = module.w_quant.z_param.data.cpu()
            if module.a_quant is not None:
                quant_params[f"{name}.a_quant.lambda_param"] = module.a_quant.lambda_param.data.cpu()
                quant_params[f"{name}.a_quant.z_param"] = module.a_quant.z_param.data.cpu()
    
    quant_path = os.path.join(save_dir, f"quant_params_epoch{epoch}.pt")
    torch.save(quant_params, quant_path)
    print(f"Quantization parameters saved to {quant_path}")

@torch.no_grad()    
def collate_for_lm(batch, tokenizer, max_length=None):
    # Remove samples where input_ids is empty
    batch = [item for item in batch if len(item["input_ids"]) > 0]

    if len(batch) == 0:
        return None

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True,
        padding_value=0
    )

    # Labels are just input_ids but pad-to-ignore
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
    learnable_modulus: bool = True
    init_lambda: float = 1.0
    init_z: float = 8.0
    per_channel: bool = False  # New option: False for per-tensor, True for per-channel
    calibrate_batches: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./saved_models"
    log_quant_params_freq: int = 100  # Log quantization params every N steps

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
    model.to(device)

    # load dataset
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    def tokenize_fn(examples):
        out = tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)
        return out
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

    # Replace Linear with QuantLinear
    quant_mode = "per-channel" if cfg.per_channel else "per-tensor"
    print(f"Using {quant_mode} quantization")
    replace_linear_with_quantlinear(
        model,
        quantize_input=cfg.quantize_input,
        learnable_modulus=cfg.learnable_modulus,
        init_lambda=cfg.init_lambda,
        init_z=cfg.init_z,
        per_channel=cfg.per_channel,
        skip_keys=[],
    )

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

    # calibration pass
    print(f"Calibrating {quant_mode} modulus parameters...")
    calibrate_model_modulus(model, val_loader, device, num_batches=cfg.calibrate_batches)

    # Log initial quantization parameters
    print("\nInitial quantization parameters after calibration:")
    log_quantization_params(model, step=0, verbose=True)

    model.train()
    print(f"Trainable parameters: {count_parameters(model)}")
    step = 0
    best_ppl = float("inf")

    # wandb
    wandb.init(project=f"modulo-quantization-{quant_mode}", config=vars(cfg))
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
                
                # Log quantization parameters periodically
                if step % cfg.log_quant_params_freq == 0:
                    log_quantization_params(model, step=step, verbose=False)
                    
            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "step": step})

        # evaluation
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
        
        # Log quantization parameters at end of epoch
        print(f"\nQuantization parameters after epoch {epoch+1}:")
        log_quantization_params(model, step=step, verbose=True)
        
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "ppl": ppl, "step": step})
        
        # Save model if best
        if ppl < best_ppl:
            best_ppl = ppl
            save_model(model, tokenizer, cfg.save_dir, epoch+1, ppl)
        
        # Save checkpoint every epoch
        save_model(model, tokenizer, os.path.join(cfg.save_dir, "checkpoints"), epoch+1, ppl)
        
        model.train()

    print(f"\nBest val PPL: {best_ppl}")
    wandb.log({"best_val_ppl": best_ppl})
    
    # Save final model
    save_model(model, tokenizer, os.path.join(cfg.save_dir, "final"), cfg.num_epochs, ppl)
    
    wandb.finish()

if __name__ == "__main__":
    main()