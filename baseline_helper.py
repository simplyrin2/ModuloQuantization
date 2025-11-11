import os
import math
import argparse
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from baseline import (
    set_seed,
    count_parameters,
    AutoTokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    load_dataset,
    replace_linear_with_scalequantlinear,
    calibrate_model_scale,
    log_scale_params,
    save_model,
    collate_for_lm,
    TrainConfig,
)


def run(cfg: TrainConfig, mode: str):
    assert mode in ("qat", "fp32", "both")
    device = torch.device(cfg.device)

    def run_qat(tokenizer, model, train_loader, val_loader):
        model.train()
        step = 0
        best_ppl = float("inf")

        wandb.init(project="baseline-qAT", config=asdict(cfg))
        wandb.watch(model, log='parameters', log_freq=100)

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
        num_warmup = int(cfg.warmup_ratio * total_steps) if total_steps > 0 else 0
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

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

                    if mode=='qat' and (step % cfg.log_scale_params_freq == 0):
                        stats = log_scale_params(model, step=step, verbose=False)
                        wandb.log(stats, step=step)

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
            avg_val_loss = float("nan") if n_batches == 0 else (val_loss / n_batches)
            ppl = float("nan") if n_batches == 0 else math.exp(avg_val_loss)
            print(f"\nEpoch {epoch+1} | train_loss={epoch_loss/len(train_loader):.4f} | val_loss={avg_val_loss:.4f} | ppl={ppl:.4f}")

            if use_qat:
                print(f"\nScale parameters after epoch {epoch+1}:")
                stats = log_scale_params(model, step=step, verbose=True)
                wandb.log(stats, step=step)

            wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "ppl": ppl, "step": step})

            if n_batches > 0 and ppl < best_ppl:
                best_ppl = ppl
                save_model(model, tokenizer, cfg.save_dir, epoch+1, ppl)

            # checkpoint every epoch
            save_model(model, tokenizer, os.path.join(cfg.save_dir, "checkpoints"), epoch+1, ppl)
            model.train()

        print(f"\nBest val PPL: {best_ppl}")
        wandb.log({"best_val_ppl": best_ppl})
        save_model(model, tokenizer, os.path.join(cfg.save_dir, "final"), cfg.num_epochs, ppl)
        wandb.finish()

    def run_fp32_eval(tokenizer, model, val_loader):
        # Evaluate pretrained FP32 model on the same evaluation set
        model.eval()
        model.to(device)

        wandb.init(project="baseline-fp32-eval", config=asdict(cfg))
        wandb.watch(model, log='parameters', log_freq=100)

        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="FP32 Eval"):
                if batch is None:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                n_batches += 1

        avg_val_loss = float("nan") if n_batches == 0 else (val_loss / n_batches)
        ppl = float("nan") if n_batches == 0 else math.exp(avg_val_loss)
        print(f"FP32 Eval | val_loss={avg_val_loss:.4f} | ppl={ppl:.4f}")
        wandb.log({"fp32_val_loss": avg_val_loss, "fp32_ppl": ppl})
        wandb.finish()
        return avg_val_loss, ppl

    def run_once(use_qat: bool):
        set_seed(cfg.seed)

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

        train_loader = DataLoader(train_dataset, batch_size=cfg.per_device_train_batch_size,
                                  shuffle=True, collate_fn=lambda b: collate_for_lm(b, tokenizer))
        val_loader = DataLoader(val_dataset, batch_size=cfg.per_device_eval_batch_size,
                                shuffle=False, collate_fn=lambda b: collate_for_lm(b, tokenizer))

        if use_qat:
            # apply QAT wrappers, move model, calibrate and train
            replace_linear_with_scalequantlinear(
                model,
                quantize_input=cfg.quantize_input,
                learnable_scale=cfg.learnable_scale,
                init_scale=cfg.init_scale,
                per_channel=cfg.per_channel,
                bits=cfg.bits,
                skip_keys=[],
            )
            # move model to device after replacement
            model.to(device)

            # calibrate if QAT
            print("Calibrating scales from validation batches...")
            calibrate_model_scale(model, val_loader, device, num_batches=cfg.calibrate_batches)
            print("\nInitial scale parameters after calibration:")
            log_scale_params(model, step=0, verbose=True)

            run_qat(tokenizer, model, train_loader, val_loader)
        else:
            # FP32: do not replace linears or train, just evaluate pretrained model
            model.to(device)
            run_fp32_eval(tokenizer, model, val_loader)

    if mode == "qat":
        run_once(use_qat=True)
    elif mode == "fp32":
        run_once(use_qat=False)
    else:  # both
        run_once(use_qat=True)
        run_once(use_qat=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qat", "fp32", "both"], default="both")
    args = parser.parse_args()
    cfg = TrainConfig()
    run(cfg, args.mode)