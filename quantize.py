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
# ModuloQuantizer (unchanged)
# --------------------------
class ModuloQuantizer(nn.Module):
    def __init__(self, init_lambda: float = 1.0, init_z: float = 8.0, learnable: bool = True):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda))) if learnable else nn.Parameter(torch.tensor(float(init_lambda)), requires_grad=False)
        self.z_param = nn.Parameter(torch.tensor(float(init_z))) if learnable else nn.Parameter(torch.tensor(float(init_z)), requires_grad=False)

    @property
    def modulus(self):
        return torch.clamp(self.z_param * self.lambda_param, min=1e-6)

    def forward(self, x):
        return modulo_ste(x, self.modulus)

# --------------------------
# QuantLinear wrapper (weights/activations)
# --------------------------
class QuantLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, quantize_input: bool = False, learnable_modulus: bool = True, init_lambda: float = 1.0, init_z: float = 8.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias_flag = base_linear.bias is not None

        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        if self.bias_flag:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.w_quant = ModuloQuantizer(init_lambda=init_lambda, init_z=init_z, learnable=learnable_modulus)
        self.a_quant = ModuloQuantizer(init_lambda=init_lambda, init_z=init_z, learnable=learnable_modulus) if quantize_input else None

        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.a_quant is not None:
            x_q = self.a_quant(x)
        else:
            x_q = x

        w_q = self.w_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def calibrate_modulus_from_data(self, xs: List[torch.Tensor], percentile: float = 0.999):
        if self.a_quant is not None and len(xs) > 0:
            cat = torch.cat([t.detach().reshape(-1) for t in xs], dim=0).abs()
            k = max(1, int(math.ceil(percentile * cat.numel())))
            topk, _ = torch.kthvalue(cat, k)
            new_mod = max(2.0 * float(topk), 1e-3)
            self.a_quant.z_param.data.fill_(1.0)
            self.a_quant.lambda_param.data.fill_(new_mod)

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
                                    skip_keys: Optional[List[str]] = None):
    if skip_keys is None:
        skip_keys = []

    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear) and all(kw not in full_name for kw in skip_keys):
            setattr(module, name, QuantLinear(child, quantize_input, learnable_modulus, init_lambda, init_z))
        else:
            replace_linear_with_quantlinear(child, quantize_input, learnable_modulus, init_lambda, init_z, skip_keys)

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
        # batch is dict with input_ids and attention_mask and labels (we'll send to device inside model call)
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)

    for h in handles:
        h.remove()

    for m in model.modules():
        if isinstance(m, QuantLinear):
            xs = activations.get(id(m), [])
            m.calibrate_modulus_from_data(xs, percentile=0.999)

    model.train()

# # --------------------------
# # Data utilities for causal LM (pad + labels)
# # --------------------------
# def collate_for_lm(batch, tokenizer, max_length=None):
#     # batch: list of dicts from dataset containing 'input_ids' etc
#     input_ids = [b["input_ids"] for b in batch]
#     attention_mask = [b["attention_mask"] for b in batch]

#     # pad to max length in batch or provided max_length
#     if max_length is None:
#         max_len = max([len(x) for x in input_ids])
#     else:
#         max_len = max_length

#     padded_ids = []
#     padded_attns = []
#     labels = []

#     for ids, attn in zip(input_ids, attention_mask):
#         # trim if needed
#         ids = ids[:max_len]
#         attn = attn[:max_len]

#         pad_len = max_len - ids.size(0)

#         # padding tensors
#         pad_ids = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
#         pad_attn = torch.zeros((pad_len,), dtype=attn.dtype)

#         # clamp to fixed length
#         padded = torch.cat([ids, pad_ids], dim=0)
#         padded_attn = torch.cat([attn, pad_attn], dim=0)

#         # labels = copy of padded input ids
#         label = padded.clone()
#         if pad_len > 0:
#             label[-pad_len:] = torch.full((pad_len,), -100, dtype=label.dtype)

#         padded_ids.append(padded)
#         padded_attns.append(padded_attn)
#         labels.append(label)

#     # stack into batch tensors
#     padded_ids = torch.stack(padded_ids, dim=0)
#     padded_attns = torch.stack(padded_attns, dim=0)
#     labels = torch.stack(labels, dim=0)

#     batch_t = {
#         "input_ids": torch.tensor(padded_ids, dtype=torch.long),
#         "attention_mask": torch.tensor(padded_attns, dtype=torch.long),
#         "labels": torch.tensor(labels, dtype=torch.long),
#     }
#     return batch_t

# @torch.no_grad()
# def collate_for_lm(batch, tokenizer, max_length=None):
#     input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
#     attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

#     # Labels are a clone of input_ids for LM loss
#     labels = [ids.clone() for ids in input_ids]

#     # Pad sequences in list form first
#     input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#     attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

#     # Pad labels and mark paddings as -100
#     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_masks,
#         "labels": labels,
#     }

@torch.no_grad()    
def collate_for_lm(batch, tokenizer, max_length=None):
    # Remove samples where input_ids is empty
    batch = [item for item in batch if len(item["input_ids"]) > 0]

    if len(batch) == 0:
        return None  # dataloader will skip

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True,
                                          padding_value=0)

    # Labels are just input_ids but pad-to-ignore
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100   # ignore loss on pad

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
    model_name: str = "gpt2"  # small baseline GPT-2
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"  # WikiText-2 raw
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
    calibrate_batches: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Main
# --------------------------
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # Ensure pad token exists for GPT2
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # load dataset (wikitext-2 raw)
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    def tokenize_fn(examples):
        # use 'text' field for wikitext raw
        out = tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)
        return out
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")

    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]

    # small collator wrapper
    def collate_fn(batch):
        return collate_for_lm(batch, tokenizer, max_length=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Replace Linear w/ QuantLinear
    replace_linear_with_quantlinear(
        model,
        quantize_input=cfg.quantize_input,
        learnable_modulus=cfg.learnable_modulus,
        init_lambda=cfg.init_lambda,
        init_z=cfg.init_z,
        skip_keys=[],
    )

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)

    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
    num_warmup = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # calibration pass
    calibrate_model_modulus(model, val_loader, device, num_batches=cfg.calibrate_batches)

    model.train()
    print(f"Trainable parameters: {count_parameters(model)}")
    step = 0
    best_ppl = float("inf")

    # wandb
    wandb.init(project="modulo-quantization-ppl", config=vars(cfg))
    wandb.watch(model, log='parameters', log_freq=100)

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch is None:
                continue  # skip empty batch
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)  # labels included in batch -> model returns loss
            loss = outputs.loss
            (loss / cfg.grad_accum_steps).backward()
            if (i + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "step": step})

        # eval: compute average loss over val set and perplexity = exp(loss)
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue  # skip empty batch
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
        print(f"Epoch {epoch+1} | train_loss={epoch_loss/len(train_loader):.4f} | val_loss={avg_val_loss:.4f} | ppl={ppl:.4f}")
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "ppl": ppl, "step": step})
        if ppl < best_ppl:
            best_ppl = ppl
        model.train()

    print("Best val PPL:", best_ppl)
    wandb.log({"best_val_ppl": best_ppl})
    wandb.finish()

if __name__ == "__main__":
    main()
