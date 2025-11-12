# full_script_modulo_qat_gpt2_ppl.py
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# Fake-quant STE helpers
# --------------------------
class RoundSTEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through for rounding
        return grad_output


def round_ste(x):
    return RoundSTEFn.apply(x)


class ModuloFoldSTEFn(torch.autograd.Function):
    """
    Fold x into [-λ, λ) with period 2λ via STE:
      forward: (x + λ) mod (2λ) �~H~R λ
      backward: pass-through w.r.t. x, no grad to λ
    """
    @staticmethod
    def forward(ctx, x, lam):
        lam = torch.clamp(lam, min=1e-8)
        y = torch.remainder(x + lam, 2.0 * lam) - lam
        quotient = torch.floor((x + lam) / (2.0 * lam))
        return y, quotient

    @staticmethod
    def backward(ctx, grad_y, grad_quotient):
        return grad_y, grad_quotient


def modulo_fold_ste(x, lam: torch.Tensor):
    return ModuloFoldSTEFn.apply(x, lam)


# --------------------------
# ModuloQuantizer - Supports both per-tensor and per-channel
# --------------------------
class ModuloQuantizer(nn.Module):
    """Modulo fake-quantizer. Only λ is learnable; folding period is 2λ."""
    def __init__(self, num_channels: Optional[int] = None, init_lambda: float = 1.0,
                 learnable: bool = True, per_channel: bool = False, bits: int = 8):
        super().__init__()
        self.per_channel = per_channel
        self.num_channels = num_channels if per_channel else 1
        self.bits = int(bits)
        if per_channel and num_channels is None:
            raise ValueError("num_channels must be specified when per_channel=True")
        shape = (self.num_channels,) if per_channel else ()
        self.lambda_param = nn.Parameter(
            torch.full(shape, float(init_lambda)),
            requires_grad=learnable
        )

    @property
    def lam(self):
        """Half-range λ used for periodic folding and uniform quant."""
        return torch.clamp(self.lambda_param, min=1e-6)

    @property
    def qmax(self) -> float:
        """Q = 2^(b-1) - 1"""
        return float((1 << (self.bits - 1)) - 1)

    def _broadcast(self, t: torch.Tensor, x: torch.Tensor, dim: Optional[int]):
        if not self.per_channel:
            return t
        if dim is None:
            raise ValueError("dim must be specified in per-channel mode")
        shape = [1] * x.ndim
        shape[dim] = self.num_channels
        return t.view(*shape)

    def forward(self, x, dim: Optional[int] = None):
        """
        Periodic folding + uniform fake-quantization with STE:
          1) w�~C = (x + λ) mod (2λ) �~H~R λ       in [-λ, λ)
          2) S = λ / Q, Q = 2^(b-1) �~H~R 1
          3) q = clip(round(w�~C / S), �~H~RQ, Q)
          4) ŵ = S · q
        Returns ŵ (dequantized), with STE for modulo and rounding.
        """
        lam = self._broadcast(self.lam, x, dim)
        w_tilde, quotient = modulo_fold_ste(x, lam)                 # [-λ, λ)
        S = lam / self.qmax
        q = torch.clamp(round_ste(w_tilde / S), min=-self.qmax, max=self.qmax)
        w_hat = S * q + quotient * (2.0 * lam)
        return w_hat


# --------------------------
# QuantLinear wrapper (weights/activations)
# --------------------------
class QuantLinear(nn.Module):
    """Linear layer with modulo fake quantization (per-tensor or per-channel)."""
    def __init__(self, base_linear: nn.Linear, quantize_input: bool = False,
                 learnable_modulus: bool = True, init_lambda: float = 1.0,
                 per_channel: bool = False, bits: int = 8):
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

        # Weight quantizer
        self.w_quant = ModuloQuantizer(
            num_channels=self.out_features if per_channel else None,
            init_lambda=init_lambda,
            learnable=learnable_modulus,
            per_channel=per_channel,
            bits=bits,
        )

        # Activation quantizer (optional)
        self.a_quant = ModuloQuantizer(
            num_channels=self.in_features if per_channel else None,
            init_lambda=init_lambda,
            learnable=learnable_modulus,
            per_channel=per_channel,
            bits=bits,
        ) if quantize_input else None

        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Fake-quant input activations
        if self.a_quant is not None:
            x_q = self.a_quant(x, dim=-1) if self.per_channel else self.a_quant(x)
        else:
            x_q = x

        # Fake-quant weights
        w_q = self.w_quant(self.weight, dim=0) if self.per_channel else self.w_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def calibrate_modulus_from_data(self, xs: List[torch.Tensor], percentile: float = 0.25):
        """Calibrate λ from data statistics (per-tensor or per-channel).
           We set λ �~I~H percentile(|x|)."""
        # Activations
        if self.a_quant is not None and len(xs) > 0:
            if self.per_channel:
                cat = torch.cat([t.detach().reshape(-1, self.in_features) for t in xs], dim=0).abs()
                for c in range(self.in_features):
                    vals = cat[:, c].reshape(-1)
                    k = max(1, int(math.ceil(percentile * vals.numel())))
                    topk, _ = torch.kthvalue(vals, k)
                    self.a_quant.lambda_param.data[c] = max(float(topk), 1e-3)
            else:
                cat = torch.cat([t.detach().reshape(-1) for t in xs], dim=0).abs()
                k = max(1, int(math.ceil(percentile * cat.numel())))
                topk, _ = torch.kthvalue(cat, k)
                self.a_quant.lambda_param.data.fill_(max(float(topk), 1e-3))
        # Weights
        if self.per_channel:
            wabs = self.weight.detach().abs()  # [out_features, in_features]
            for c in range(self.out_features):
                vals = wabs[c, :].reshape(-1)
                k2 = max(1, int(math.ceil(percentile * vals.numel())))
                topk2, _ = torch.kthvalue(vals, k2)
                self.w_quant.lambda_param.data[c] = max(float(topk2), 1e-3)
        else:
            wabs = self.weight.detach().abs().reshape(-1)
            k2 = max(1, int(math.ceil(percentile * wabs.numel())))
            topk2, _ = torch.kthvalue(wabs, k2)
            self.w_quant.lambda_param.data.fill_(max(float(topk2), 1e-3))

        self.calibrated.data.fill_(1)


# --------------------------
# Replace Linear with QuantLinear in model
# --------------------------
def replace_linear_with_quantlinear(module: nn.Module,
                                    quantize_input: bool = False,
                                    learnable_modulus: bool = True,
                                    init_lambda: float = 1.0,
                                    per_channel: bool = False,
                                    bits: int = 8,
                                    skip_keys: Optional[List[str]] = None):
    if skip_keys is None:
        skip_keys = []

    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear) and all(kw not in full_name for kw in skip_keys):
            setattr(module, name, QuantLinear(
                child, quantize_input, learnable_modulus,
                init_lambda, per_channel, bits
            ))
        else:
            replace_linear_with_quantlinear(
                child, quantize_input, learnable_modulus,
                init_lambda, per_channel, bits, skip_keys
            )

# Calibration pass
# --------------------------
@torch.no_grad()
def calibrate_model_modulus(model: nn.Module, loader: DataLoader, device: torch.device,
                            num_batches: int = 20, percentile: float = 0.999):
    """
    Run a short eval pass to collect activations per QuantLinear and calibrate λ.
    """
    model.eval()
    inp_buffers = {}

    # Collect inputs via pre-forward hooks
    def make_hook(mod):
        def hook(mod, inputs):
            x = inputs[0]
            if mod not in inp_buffers:
                inp_buffers[mod] = []
            # keep small sample to limit memory
            if x.dim() >= 2:
                # take at most 2 sequences to reduce memory
                xs = x[:2].detach().cpu()
            else:
                xs = x.detach().cpu()
            inp_buffers[mod].append(xs)
        return hook

    handles = []
    for m in model.modules():
        if isinstance(m, QuantLinear):
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

    # Calibrate each QuantLinear
    for m in model.modules():
        if isinstance(m, QuantLinear):
            xs = inp_buffers.get(m, [])
            m.calibrate_modulus_from_data(xs, percentile=percentile)           

# --------------------------
# Log quantization parameters
# --------------------------
@torch.no_grad()
def log_quantization_params(model: nn.Module, step: int, verbose: bool = True):
    """Log λ stats to wandb and optionally print them."""
    quant_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            w_lambda = module.w_quant.lambda_param.data
            w_lam = module.w_quant.lam.data
            quant_stats[f"{name}/w_lambda_mean"] = w_lambda.mean().item()
            quant_stats[f"{name}/w_lambda_std"] = w_lambda.std().item() if w_lambda.numel() > 1 else 0.0
            quant_stats[f"{name}/w_lambda_min"] = w_lambda.min().item()
            quant_stats[f"{name}/w_lambda_max"] = w_lambda.max().item()
            quant_stats[f"{name}/w_modulus_mean"] = w_lam.mean().item()
            quant_stats[f"{name}/w_modulus_min"] = w_lam.min().item()
            quant_stats[f"{name}/w_modulus_max"] = w_lam.max().item()

            if module.a_quant is not None:
                a_lambda = module.a_quant.lambda_param.data
                a_lam = module.a_quant.lam.data
                quant_stats[f"{name}/a_lambda_mean"] = a_lambda.mean().item()
                quant_stats[f"{name}/a_lambda_std"] = a_lambda.std().item() if a_lambda.numel() > 1 else 0.0
                quant_stats[f"{name}/a_lambda_min"] = a_lambda.min().item()
                quant_stats[f"{name}/a_lambda_max"] = a_lambda.max().item()
                quant_stats[f"{name}/a_modulus_mean"] = a_lam.mean().item()
                quant_stats[f"{name}/a_modulus_min"] = a_lam.min().item()
                quant_stats[f"{name}/a_modulus_max"] = a_lam.max().item()

            if verbose:
                print(f"\n{name}:")
                print(f"  Weight - lambda: {w_lambda.mean().item():.4f} ± {w_lambda.std().item():.4f}, "
                      f"λ(mean): {w_lam.mean().item():.4f}")
                if module.a_quant is not None:
                    print(f"  Activation - lambda: {a_lambda.mean().item():.4f} ± {a_lambda.std().item():.4f}, "
                          f"λ(mean): {a_lam.mean().item():.4f}")

    # wandb.log(quant_stats, step=step)
    return quant_stats

# --------------------------
# Log quantization parameters
# --------------------------
@torch.no_grad()
def log_quantization_params(model: nn.Module, step: int, verbose: bool = True):
    """Log λ stats to wandb and optionally print them."""
    quant_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            w_lambda = module.w_quant.lambda_param.data
            w_lam = module.w_quant.lam.data
            quant_stats[f"{name}/w_lambda_mean"] = w_lambda.mean().item()
            quant_stats[f"{name}/w_lambda_std"] = w_lambda.std().item() if w_lambda.numel() > 1 else 0.0
            quant_stats[f"{name}/w_lambda_min"] = w_lambda.min().item()
            quant_stats[f"{name}/w_lambda_max"] = w_lambda.max().item()
            quant_stats[f"{name}/w_modulus_mean"] = w_lam.mean().item()
            quant_stats[f"{name}/w_modulus_min"] = w_lam.min().item()
            quant_stats[f"{name}/w_modulus_max"] = w_lam.max().item()

            if module.a_quant is not None:
                a_lambda = module.a_quant.lambda_param.data
                a_lam = module.a_quant.lam.data
                quant_stats[f"{name}/a_lambda_mean"] = a_lambda.mean().item()
                quant_stats[f"{name}/a_lambda_std"] = a_lambda.std().item() if a_lambda.numel() > 1 else 0.0
                quant_stats[f"{name}/a_lambda_min"] = a_lambda.min().item()
                quant_stats[f"{name}/a_lambda_max"] = a_lambda.max().item()
                quant_stats[f"{name}/a_modulus_mean"] = a_lam.mean().item()
                quant_stats[f"{name}/a_modulus_min"] = a_lam.min().item()
                quant_stats[f"{name}/a_modulus_max"] = a_lam.max().item()

            if verbose:
                print(f"\n{name}:")
                print(f"  Weight - lambda: {w_lambda.mean().item():.4f} ± {w_lambda.std().item():.4f}, "
                      f"λ(mean): {w_lam.mean().item():.4f}")
                if module.a_quant is not None:
                    print(f"  Activation - lambda: {a_lambda.mean().item():.4f} ± {a_lambda.std().item():.4f}, "
                          f"λ(mean): {a_lam.mean().item():.4f}")

    # wandb.log(quant_stats, step=step)
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
            if module.a_quant is not None:
                quant_params[f"{name}.a_quant.lambda_param"] = module.a_quant.lambda_param.data.cpu()

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
    
    
# Weight distribution plotting

@torch.no_grad()
def _compute_weight_quant_triplet(m: "QuantLinear"):
    """
    For a QuantLinear with ModuloQuantizer:
      - float weights: w
      - integer quant: q = clip(round(w_tilde / S), -qmax, qmax)
      - dequantized:  w_hat = S*q + quotient*(2λ)
    """
    # weights
    w = m.weight.detach().cpu()  # [out, in]

    # λ and qmax
    lam = m.w_quant.lam.detach().cpu()  # () or [out]
    qmax = m.w_quant.qmax

    # broadcast λ per out channel if needed
    if m.per_channel and lam.numel() > 1:
        lam_b = lam.view(-1, 1)  # [out,1]
    else:
        lam_b = lam  # scalar

    # Fold into [-λ, λ): w_tilde, and count periods with quotient
    w_tilde = torch.remainder(w + lam_b, 2.0 * lam_b) - lam_b
    quotient = torch.floor((w + lam_b) / (2.0 * lam_b))

    # Uniform quant scale
    S = lam_b / qmax
    q = torch.clamp(torch.round(w_tilde / S), min=-qmax, max=qmax)
    # Dequant following your forward()
    w_hat = S * q + quotient * (2.0 * lam_b)
    return w, q, w_hat


@torch.no_grad()
def plot_weight_distributions(model: nn.Module, out_dir: str, epoch: int):
    os.makedirs(out_dir, exist_ok=True)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            try:
                w, q, w_hat = _compute_weight_quant_triplet(module)
                safe_name = name.replace(".", "_")

                # Plot 1: float W vs dequant Ŵ
                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
                ax1.hist(w.flatten().numpy(), bins=100, alpha=0.8, label="float W")
                ax1.hist(w_hat.flatten().numpy(), bins=100, alpha=0.6, label="dequant Ŵ")
                ax1.set_title(f"{name} float vs dequant (epoch {epoch})")
                ax1.set_xlabel("value")
                ax1.set_ylabel("count")
                ax1.legend()
                fig1.tight_layout()
                fig1.savefig(os.path.join(out_dir, f"{safe_name}_float_dequant_epoch{epoch}.png"), dpi=150)
                plt.close(fig1)

                # Plot 2: integer Q distribution (separate)
                fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
                qmax = module.w_quant.qmax
                q_bins = np.arange(-qmax - 0.5, qmax + 1.5, 1.0)
                ax2.hist(q.flatten().numpy(), bins=q_bins, alpha=0.8, color="tab:green")
                ax2.set_title(f"{name} integer Q (epoch {epoch})")
                ax2.set_xlabel("integer level")
                ax2.set_ylabel("count")
                fig2.tight_layout()
                fig2.savefig(os.path.join(out_dir, f"{safe_name}_intQ_epoch{epoch}.png"), dpi=150)
                plt.close(fig2)

            except Exception as e:
                print(f"[plot_weight_distributions] Skipped {name}: {e}")


@torch.no_grad()
def dump_and_plot_scales(model: nn.Module, out_dir: str, epoch: int, filename: str = None):
    """
    Dump λ (lambda) values and plot their distributions (weights and activations separately).
    """
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = os.path.join(out_dir, f"lambda_values_epoch{epoch}.txt")

    lines = []
    all_w_lams = []
    all_a_lams = []

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # Weight lambdas
            lam_w = module.w_quant.lambda_param.detach().cpu().view(-1)
            lines.append(f"{name}.w_lambda: " + " ".join(f"{v:.8g}" for v in lam_w.tolist()))
            if lam_w.numel() > 0:
                all_w_lams.append(lam_w)

            # Activation lambdas (if present)
            if module.a_quant is not None:
                lam_a = module.a_quant.lambda_param.detach().cpu().view(-1)
                lines.append(f"{name}.a_lambda: " + " ".join(f"{v:.8g}" for v in lam_a.tolist()))
                if lam_a.numel() > 0:
                    all_a_lams.append(lam_a)

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Lambda values written to {filename}")

    # Plot aggregated histograms
    if len(all_w_lams) > 0:
        wcat = torch.cat(all_w_lams).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.hist(wcat, bins=100, alpha=0.8, color="tab:blue")
        ax.set_title(f"Weight λ distribution (epoch {epoch})")
        ax.set_xlabel("λ")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lambda_weights_hist_epoch{epoch}.png"), dpi=150)
        plt.close(fig)

    if len(all_a_lams) > 0:
        acat = torch.cat(all_a_lams).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.hist(acat, bins=100, alpha=0.8, color="tab:orange")
        ax.set_title(f"Activation λ distribution (epoch {epoch})")
        ax.set_xlabel("λ")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lambda_activations_hist_epoch{epoch}.png"), dpi=150)
        plt.close(fig)
        
@torch.no_grad()
def plot_weight_channel_distributions(
    model: nn.Module,
    out_dir: str,
    epoch: int,
    channels_per_layer: int = 4,
    channel_indices=None,  # optional dict: {layer_name: [idx0, idx1, ...]}
):
    """
    For per-channel quant, plot per-output-channel distributions:
      - Overlay float W_c and dequant Ŵ_c in one plot, with vertical lines at ±λ_c
      - Plot integer Q_c in a separate plot
    Falls back to a single global λ if per-tensor quant is used.
    """
    out_dir = os.path.join(out_dir, "channels")
    os.makedirs(out_dir, exist_ok=True)

    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        W = module.weight.detach().cpu()  # [out, in]
        out_ch, in_ch = W.shape
        safe_name = name.replace(".", "_")

        # Determine indices to plot
        if channel_indices and name in channel_indices:
            idxs = [i for i in channel_indices[name] if 0 <= i < out_ch]
        else:
            # pick evenly spaced channels
            if channels_per_layer >= out_ch:
                idxs = list(range(out_ch))
            else:
                idxs = np.linspace(0, out_ch - 1, channels_per_layer, dtype=int).tolist()

        # Prepare lambdas (per-channel or scalar)
        lam_param = module.w_quant.lam.detach().cpu()
        per_channel = module.per_channel and lam_param.numel() > 1
        qmax = module.w_quant.qmax

        for c in idxs:
            w_c = W[c, :]  # [in]
            lam_c = lam_param[c].item() if per_channel else float(lam_param.item())

            # Compute folded, int q, and dequant for this channel
            lam_t = torch.tensor(lam_c)
            w_tilde = torch.remainder(w_c + lam_t, 2.0 * lam_t) - lam_t
            quotient = torch.floor((w_c + lam_t) / (2.0 * lam_t))
            S = lam_t / qmax
            q_c = torch.clamp(torch.round(w_tilde / S), min=-qmax, max=qmax)
            w_hat_c = S * q_c + quotient * (2.0 * lam_t)

            # Plot 1: float vs dequant with ±λ lines
            fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
            ax1.hist(w_c.numpy(), bins=100, alpha=0.8, label="float W[c]")
            ax1.hist(w_hat_c.numpy(), bins=100, alpha=0.6, label="dequant Ŵ[c]")
            ax1.axvline(-lam_c, color="red", linestyle="--", linewidth=1.5, label="-λ")
            ax1.axvline(+lam_c, color="red", linestyle="--", linewidth=1.5, label="+λ")
            ax1.set_title(f"{name} ch {c} (λ={lam_c:.4g}) | float vs dequant (epoch {epoch})")
            ax1.set_xlabel("value")
            ax1.set_ylabel("count")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(os.path.join(out_dir, f"{safe_name}_ch{c}_float_dequant_epoch{epoch}.png"), dpi=150)
            plt.close(fig1)

            # Plot 2: integer Q for this channel
            fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
            q_bins = np.arange(-qmax - 0.5, qmax + 1.5, 1.0)
            ax2.hist(q_c.numpy(), bins=q_bins, alpha=0.85, color="tab:green")
            ax2.set_title(f"{name} ch {c} integer Q (epoch {epoch})")
            ax2.set_xlabel("integer level")
            ax2.set_ylabel("count")
            fig2.tight_layout()
            fig2.savefig(os.path.join(out_dir, f"{safe_name}_ch{c}_intQ_epoch{epoch}.png"), dpi=150)
            plt.close(fig2)

def evaluate_model(model):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    from tqdm import tqdm
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    device = next(model.parameters()).device
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return avg_nll, ppl

    # print(f"Test PPL: {ppl.item():.2f}")


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
    num_epochs: int = 3
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 1
    seed: int = 42
    quantize_input: bool = False
    learnable_modulus: bool = True
    init_lambda: float = 1.0
    per_channel: bool = True
    calibrate_batches: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./saved_models"
    log_quant_params_freq: int = 100
    bits: int = 8  # number of quantization bits
    percentile: float = 0.999  # for calibration

# --------------------------


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
    # model.to(device)

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
        per_channel=cfg.per_channel,
        bits=cfg.bits,
        skip_keys=[],
    )
    model.to(device)

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
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

    analysis_dir = os.path.join(cfg.save_dir, "analysis4")
    dump_and_plot_scales(model, analysis_dir, epoch=0)
    plot_weight_distributions(model, analysis_dir, epoch=0)
    plot_weight_channel_distributions(model, analysis_dir, epoch=0, channels_per_layer=4)

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
        # avg_val_loss, ppl = evaluate_model(model)

        print(f"\nEpoch {epoch+1} | train_loss={epoch_loss/len(train_loader):.4f} | val_loss={avg_val_loss:.4f} | ppl={ppl:.4f}")

        # Log quantization parameters at end of epoch
        print(f"\nQuantization parameters after epoch {epoch+1}:")
        quant_stats = log_quantization_params(model, step=step, verbose=True)

        wandb.log(quant_stats)

        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "ppl": ppl, "step": step})
        
        dump_and_plot_scales(model, analysis_dir, epoch=epoch+1)
        plot_weight_distributions(model, analysis_dir, epoch=epoch+1)
        plot_weight_channel_distributions(model, analysis_dir, epoch=epoch+1, channels_per_layer=4)

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