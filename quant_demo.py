#!/usr/bin/env python3
import argparse
import math
from typing import Tuple, Optional, List

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe by default
import matplotlib.pyplot as plt


def modulo_fake_quantize(x: torch.Tensor, lam: torch.Tensor, bits: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements:
      w̃ = (x + λ) mod (2λ) − λ
      Q = 2^(b−1) − 1, S = λ / Q
      q = clip(round(w̃ / S), −Q, Q)
      ŵ = S · q
    Returns: w_tilde, q, w_hat, S
    """
    lam = torch.clamp(lam, min=1e-6)
    qmax = float((1 << (int(bits) - 1)) - 1)
    w_tilde = torch.remainder(x + lam, 2.0 * lam) - lam
    # find also the quotient after division by 2λ
    quotient = torch.floor((x + lam) / (2.0 * lam))
    S = lam / qmax
    q = torch.clamp(torch.round(w_tilde / S), min=-qmax, max=qmax)
    w_hat = S * q + quotient * (2.0 * lam)
    return w_tilde, q, w_hat, S


def make_inputs(values: Optional[List[float]], lam: float, num_points: int, range_mul: float) -> torch.Tensor:
    if values and len(values) > 0:
        return torch.tensor(values, dtype=torch.float32)
    lo = -range_mul * lam
    hi = range_mul * lam
    return torch.linspace(lo, hi, steps=max(2, num_points), dtype=torch.float32)


def plot_results(x: torch.Tensor, w_tilde: torch.Tensor, q: torch.Tensor, w_hat: torch.Tensor,
                 lam: float, bits: int, out_path: str, show: bool = False):
    x_np = x.cpu().numpy()
    wtil_np = w_tilde.cpu().numpy()
    q_np = q.cpu().numpy()
    what_np = w_hat.cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Modulo Fake-Quantization | λ={lam:.4g}, bits={bits}", fontsize=14)

    # 1) Folding
    axs[0].plot(x_np, x_np, '--', color='gray', label='y=x')
    axs[0].plot(x_np, wtil_np, color='tab:blue', label='w̃ (folded into [-λ, λ))')
    axs[0].axhline(+lam, color='k', lw=0.7, ls=':')
    axs[0].axhline(-lam, color='k', lw=0.7, ls=':')
    axs[0].set_ylabel("w̃")
    axs[0].legend(loc="best")

    # 2) Quantized integer level
    axs[1].step(x_np, q_np, where='mid', color='tab:orange', label='q (int)')
    axs[1].set_ylabel("q")
    axs[1].legend(loc="best")

    # 3) Dequantized
    axs[2].plot(x_np, x_np, '--', color='gray', label='y=x')
    axs[2].step(x_np, what_np, where='mid', color='tab:green', label='ŵ = S·q')
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("ŵ")
    axs[2].legend(loc="best")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    if show:
        matplotlib.use("TkAgg", force=True)  # try to display if a backend exists
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize modulo fake-quantization and dequantization.")
    ap.add_argument("--lam", type=float, default=3.0, help="Lambda (half-range).")
    ap.add_argument("--bits", type=int, default=8, help="Quantization bits.")
    ap.add_argument("--num-points", type=int, default=2001, help="Points for sweep when --values not given.")
    ap.add_argument("--range-mul", type=float, default=4.0, help="Sweep range is ±range_mul*lam.")
    ap.add_argument("--values", type=float, nargs="*", help="Explicit values to test (overrides sweep).")
    ap.add_argument("--out", type=str, default="quant_demo.png", help="Output plot path.")
    ap.add_argument("--print-table", action="store_true", help="Print a small mapping table for first few points.")
    ap.add_argument("--show", action="store_true", help="Try to show the plot window.")
    args = ap.parse_args()

    x = make_inputs(args.values, args.lam, args.num_points, args.range_mul)
    print(x)
    lam_t = torch.tensor(args.lam, dtype=torch.float32)

    w_tilde, q, w_hat, S = modulo_fake_quantize(x, lam_t, args.bits)

    mse = torch.mean((x - w_hat) ** 2).item()
    mae = torch.mean(torch.abs(x - w_hat)).item()
    qmax = (1 << (args.bits - 1)) - 1

    print(f"Lambda λ = {args.lam:.6f}, bits = {args.bits}, qmax = {qmax}, S = {S.item():.6g}")
    print(f"Error: MSE = {mse:.6g}, MAE = {mae:.6g}")

    if args.print_table:
        k = min(20, x.numel())
        header = f"{'x':>12} {'w~':>12} {'q':>8} {'w^':>12} {'err':>12}"
        print(header)
        print("-" * len(header))
        for xi, wi, qi, woi in zip(x[:k], w_tilde[:k], q[:k], w_hat[:k]):
            print(f"{xi.item():12.6f} {wi.item():12.6f} {int(qi.item()):8d} {woi.item():12.6f} {(xi.item()-woi.item()):12.6f}")

    plot_results(x, w_tilde, q, w_hat, args.lam, args.bits, args.out, show=args.show)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()