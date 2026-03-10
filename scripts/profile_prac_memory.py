#!/usr/bin/env python3
"""
Profile peak GPU memory for Baseline (LoRA) vs PRAC+LoRA across batch sizes.

Runs 5 training steps per (batch_size, mode) configuration and records
torch.cuda.max_memory_allocated(). Finds the crossover batch size where
PRAC starts saving memory compared to baseline.

Usage (run both modes in parallel on separate GPUs):
    # Terminal 1 — baseline on GPU 0
    CUDA_VISIBLE_DEVICES=0 python scripts/profile_prac_memory.py --mode baseline

    # Terminal 2 — PRAC on GPU 1
    CUDA_VISIBLE_DEVICES=1 python scripts/profile_prac_memory.py --mode prac

    # Or run sequentially on one GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/profile_prac_memory.py --mode both
"""

import torch
import gc
import json
import time
import os
import sys
import argparse
import copy
import traceback

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats


LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def format_example(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    combined = f"{instruction}\n{inp}" if inp else instruction
    return {"text": LLAMA3_TEMPLATE.format(instruction=combined, output=output)}


def tokenize_with_labels(examples, tokenizer, max_length):
    all_input_ids, all_labels, all_attention_mask = [], [], []
    sep = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    for text in examples["text"]:
        parts = text.split(sep, 1)
        prompt_text = parts[0] + sep if len(parts) > 1 else ""
        tok_full = tokenizer(text, max_length=max_length, truncation=True, padding=False, return_attention_mask=False)
        input_ids = tok_full["input_ids"]
        if prompt_text:
            tok_prompt = tokenizer(prompt_text, max_length=max_length, truncation=True, padding=False, return_attention_mask=False)
            prompt_len = len(tok_prompt["input_ids"])
        else:
            prompt_len = 0
        labels = copy.deepcopy(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append([1] * len(input_ids))
    return {"input_ids": all_input_ids, "labels": all_labels, "attention_mask": all_attention_mask}


def profile_single(batch_size, use_prac, model_name, tokenizer, train_ds, args):
    """Run 5 training steps and return peak memory in MB, or -1 on OOM."""
    clear_gpu()
    torch.manual_seed(42)
    tag = "prac" if use_prac else "baseline"

    print(f"  [{tag}] batch_size={batch_size} ... ", end="", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        if use_prac:
            prac_config = PRACConfig(
                principal_rank=args.principal_rank, random_rank=args.random_rank,
                principal_update_freq=args.prac_update_freq, random_update_freq=args.prac_update_freq,
            )
            model = apply_prac_to_model(model, prac_config)

        out_dir = os.path.join(args.output_dir, f"profile_{tag}_bs{batch_size}")
        os.makedirs(out_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=out_dir,
            max_steps=5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=42,
            max_grad_norm=0.3,
            dataloader_drop_last=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
        )

        clear_gpu()
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2
        print(f"{peak_mb:.0f} MB  ({elapsed:.1f}s)")

        del model, trainer
        clear_gpu()
        return peak_mb

    except torch.cuda.OutOfMemoryError:
        print("OOM!")
        # Clean up after OOM
        try:
            del model
        except NameError:
            pass
        try:
            del trainer
        except NameError:
            pass
        clear_gpu()
        return -1

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        try:
            del model
        except NameError:
            pass
        try:
            del trainer
        except NameError:
            pass
        clear_gpu()
        return -2


def main():
    p = argparse.ArgumentParser(description="Profile PRAC memory across batch sizes")
    p.add_argument("--model_name", type=str, default="/data/models/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", type=str, default="./prac_profile")
    p.add_argument("--mode", type=str, default="both", choices=["baseline", "prac", "both"],
                   help="Which mode to profile. Use 'baseline'/'prac' for parallel GPU runs.")
    p.add_argument("--batch_sizes", type=str, default="2,4,8,16,24,32",
                   help="Comma-separated batch sizes to test")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--principal_rank", type=float, default=0.3)
    p.add_argument("--random_rank", type=float, default=0.3)
    p.add_argument("--prac_update_freq", type=int, default=200)
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_name, gpu_mem_gb = "N/A", 0

    print("=" * 70)
    print("  PRAC Memory Profiling: Llama-3.1-8B-Instruct")
    print(f"  Mode: {args.mode}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"  PRAC: r1={args.principal_rank}, r2={args.random_rank}")
    print("=" * 70)

    # Load tokenizer + dataset once
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_samples = max(batch_sizes) * 10
    print(f"\n[INFO] Loading {max_samples} Alpaca samples...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(format_example, remove_columns=ds.column_names)
    train_ds = ds.map(
        lambda ex: tokenize_with_labels(ex, tokenizer, args.max_seq_length),
        batched=True, remove_columns=ds.column_names,
    )

    # Profile
    results = {}

    if args.mode in ("baseline", "both"):
        print("\n--- Profiling BASELINE (LoRA only) ---")
        baseline_results = {}
        for bs in batch_sizes:
            peak = profile_single(bs, use_prac=False, model_name=args.model_name,
                                  tokenizer=tokenizer, train_ds=train_ds, args=args)
            baseline_results[bs] = peak
            if peak == -1:
                print(f"  Stopping baseline profiling — OOM at batch_size={bs}")
                break
        results["baseline"] = baseline_results

    if args.mode in ("prac", "both"):
        print("\n--- Profiling PRAC (LoRA + PRAC) ---")
        prac_results = {}
        for bs in batch_sizes:
            peak = profile_single(bs, use_prac=True, model_name=args.model_name,
                                  tokenizer=tokenizer, train_ds=train_ds, args=args)
            prac_results[bs] = peak
            if peak == -1:
                print(f"  Stopping PRAC profiling — OOM at batch_size={bs}")
                break
        results["prac"] = prac_results

    # Save results
    out_file = os.path.join(args.output_dir, f"profile_{args.mode}.json")
    with open(out_file, "w") as f:
        json.dump({"results": results, "gpu": gpu_name, "gpu_mem_gb": gpu_mem_gb,
                    "batch_sizes": batch_sizes, "model": args.model_name}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print summary if both modes available
    if args.mode == "both" and "baseline" in results and "prac" in results:
        print_summary(results["baseline"], results["prac"], batch_sizes)


def print_summary(baseline, prac, batch_sizes):
    print("\n\n" + "=" * 80)
    print("  MEMORY PROFILE: Baseline (LoRA) vs PRAC+LoRA")
    print("=" * 80)
    print(f"\n  {'Batch':<8} {'Baseline MB':>14} {'PRAC MB':>14} {'Delta MB':>14} {'Savings %':>12} {'Status':>10}")
    print(f"  {'-'*72}")

    crossover_bs = None
    for bs in batch_sizes:
        b = baseline.get(bs)
        p = prac.get(bs)
        if b is None or b < 0:
            b_str = "OOM" if b == -1 else "N/A"
            print(f"  {bs:<8} {b_str:>14}", end="")
        else:
            print(f"  {bs:<8} {b:>14.0f}", end="")

        if p is None or p < 0:
            p_str = "OOM" if p == -1 else "N/A"
            print(f" {p_str:>14}")
            continue
        else:
            print(f" {p:>14.0f}", end="")

        if b and b > 0 and p and p > 0:
            delta = p - b
            savings = (1 - p / b) * 100
            status = "SAVES" if savings > 0 else "OVERHEAD"
            print(f" {delta:>+14.0f} {savings:>+11.1f}% {status:>10}")
            if savings > 0 and crossover_bs is None:
                crossover_bs = bs
        else:
            print()

    print(f"\n  {'='*72}")
    if crossover_bs:
        print(f"  CROSSOVER: PRAC starts saving memory at batch_size >= {crossover_bs}")
    else:
        print(f"  CROSSOVER: Not reached in tested batch sizes. Try larger batches.")
    print("=" * 80)


def merge_and_print(output_dir, batch_sizes_str):
    """Merge baseline and prac JSON results and print summary."""
    batch_sizes = [int(x) for x in batch_sizes_str.split(",")]

    baseline_file = os.path.join(output_dir, "profile_baseline.json")
    prac_file = os.path.join(output_dir, "profile_prac.json")

    baseline, prac = {}, {}
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            data = json.load(f)
        baseline = {int(k): v for k, v in data["results"].get("baseline", {}).items()}
    if os.path.exists(prac_file):
        with open(prac_file) as f:
            data = json.load(f)
        prac = {int(k): v for k, v in data["results"].get("prac", {}).items()}

    if baseline and prac:
        print_summary(baseline, prac, batch_sizes)

        merged = {"baseline": baseline, "prac": prac}
        merged_file = os.path.join(output_dir, "profile_merged.json")
        with open(merged_file, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nMerged results saved to {merged_file}")
    else:
        print("Missing baseline or prac results. Run both modes first.")


if __name__ == "__main__":
    # Quick check: if --merge flag is passed, just merge and print
    if "--merge" in sys.argv:
        idx = sys.argv.index("--merge")
        output_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "./prac_profile"
        bs_str = "2,4,8,16,24,32"
        for i, a in enumerate(sys.argv):
            if a == "--batch_sizes" and i + 1 < len(sys.argv):
                bs_str = sys.argv[i + 1]
        merge_and_print(output_dir, bs_str)
    else:
        main()
