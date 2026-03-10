#!/usr/bin/env python3
"""
Llama-3.1-8B-Instruct benchmark: Baseline (LoRA) vs PRAC+LoRA.

Runs two fine-tuning experiments on the same Alpaca subset and compares:
  - Peak GPU memory (torch.cuda.max_memory_allocated)
  - Training loss
  - Throughput (samples/sec)
  - Wall-clock time

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_llama8b.py \
        --model_name /data/models/Llama-3.1-8B-Instruct \
        --max_train_samples 1000 --num_epochs 1
"""

import torch
import gc
import json
import time
import os
import sys
import argparse
import copy
import subprocess

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

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def nvidia_smi_mem(gpu_idx=0):
    """Get memory usage from nvidia-smi for cross-validation."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_idx)],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return -1


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


def run_experiment(tag, model_name, tokenizer, train_ds, eval_ds, args, use_prac=False):
    """Run a single fine-tuning experiment, return results dict."""
    clear_gpu()
    torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  Experiment: {tag}")
    print(f"  Model: {model_name}")
    print(f"  PRAC: {use_prac}, LoRA: True")
    print(f"  Samples: {len(train_ds)}, Epochs: {args.num_epochs}, Batch: {args.batch_size}")
    print(f"{'='*70}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    prac_info = {}
    if use_prac:
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank,
            principal_update_freq=args.prac_update_freq,
            random_update_freq=args.prac_update_freq,
        )
        model = apply_prac_to_model(model, prac_config)
        stats = get_prac_stats(model)
        enabled = [s for s in stats.values() if s["enabled"]]
        if enabled:
            avg_sav = sum(s["savings_pct"] for s in enabled) / len(enabled)
            prac_info = {"n_layers": len(enabled), "avg_savings_pct": avg_sav}
            print(f"  [PRAC] {len(enabled)} layers wrapped, avg activation savings {avg_sav:.1f}%")

    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        max_grad_norm=0.3,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )

    clear_gpu()
    mem_before_torch = torch.cuda.memory_allocated(0) / 1024**2
    mem_before_smi = nvidia_smi_mem(0)

    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    peak_mem_torch = torch.cuda.max_memory_allocated(0) / 1024**2
    peak_mem_smi = nvidia_smi_mem(0)

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    total_steps = train_result.metrics.get("train_steps", len(train_losses) * 10)
    samples_per_sec = train_result.metrics.get("train_samples_per_second", len(train_ds) * args.num_epochs / train_time)

    result = {
        "tag": tag,
        "use_prac": use_prac,
        "peak_gpu_mem_torch_mb": peak_mem_torch,
        "peak_gpu_mem_smi_mb": peak_mem_smi,
        "model_mem_before_mb": mem_before_torch,
        "train_time_s": train_time,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "samples_per_sec": samples_per_sec,
        "prac_info": prac_info,
        "train_losses": train_losses,
    }

    print(f"\n  --- {tag} Results ---")
    print(f"  Peak GPU memory (torch): {peak_mem_torch:.0f} MB")
    print(f"  Peak GPU memory (smi):   {peak_mem_smi} MB")
    print(f"  Final train loss:        {result['final_train_loss']:.4f}" if result['final_train_loss'] else "  N/A")
    print(f"  Final eval loss:         {result['final_eval_loss']:.4f}" if result['final_eval_loss'] else "  N/A")
    print(f"  Throughput:              {samples_per_sec:.2f} samples/sec")
    print(f"  Wall time:               {train_time:.1f}s")

    del model, trainer
    clear_gpu()

    return result


def main():
    p = argparse.ArgumentParser(description="Llama-3.1-8B Benchmark: Baseline vs PRAC")
    p.add_argument("--model_name", type=str, default="/data/models/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", type=str, default="./llama8b_benchmark")
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--principal_rank", type=float, default=0.3)
    p.add_argument("--random_rank", type=float, default=0.3)
    p.add_argument("--prac_update_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Llama-3.1-8B Benchmark: Baseline (LoRA) vs PRAC+LoRA")
    print(f"  Model: {args.model_name}")
    print(f"  Samples: {args.max_train_samples}, Epochs: {args.num_epochs}")
    print(f"  Batch: {args.batch_size}, Accum: {args.gradient_accumulation_steps}")
    print(f"  PRAC: r1={args.principal_rank}, r2={args.random_rank}, freq={args.prac_update_freq}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 70)

    # Load tokenizer and dataset once (shared)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[INFO] Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(min(args.max_train_samples, len(ds))))
    ds = ds.map(format_example, remove_columns=ds.column_names)

    n_eval = min(100, max(1, len(ds) // 10))
    split = ds.train_test_split(test_size=n_eval, seed=args.seed)

    print(f"[INFO] Tokenizing {len(split['train'])} train + {len(split['test'])} eval samples...")
    train_ds = split["train"].map(
        lambda ex: tokenize_with_labels(ex, tokenizer, args.max_seq_length),
        batched=True, remove_columns=split["train"].column_names,
    )
    eval_ds = split["test"].map(
        lambda ex: tokenize_with_labels(ex, tokenizer, args.max_seq_length),
        batched=True, remove_columns=split["test"].column_names,
    )

    # Run experiments
    results = []

    r_base = run_experiment("baseline_lora", args.model_name, tokenizer, train_ds, eval_ds, args, use_prac=False)
    results.append(r_base)

    r_prac = run_experiment("prac_lora", args.model_name, tokenizer, train_ds, eval_ds, args, use_prac=True)
    results.append(r_prac)

    # Summary
    print("\n\n" + "=" * 80)
    print("  SUMMARY: Llama-3.1-8B-Instruct — Baseline (LoRA) vs PRAC+LoRA")
    print("=" * 80)

    bm = r_base["peak_gpu_mem_torch_mb"]
    pm = r_prac["peak_gpu_mem_torch_mb"]
    mem_save = (1 - pm / bm) * 100 if bm > 0 else 0

    print(f"\n  {'Metric':<30} {'Baseline':>15} {'PRAC':>15} {'Delta':>15}")
    print(f"  {'-'*75}")
    print(f"  {'Peak GPU Mem (torch) MB':<30} {bm:>15.0f} {pm:>15.0f} {mem_save:>+14.1f}%")

    bm_smi = r_base["peak_gpu_mem_smi_mb"]
    pm_smi = r_prac["peak_gpu_mem_smi_mb"]
    if bm_smi > 0 and pm_smi > 0:
        smi_save = (1 - pm_smi / bm_smi) * 100
        print(f"  {'Peak GPU Mem (smi) MB':<30} {bm_smi:>15} {pm_smi:>15} {smi_save:>+14.1f}%")

    bl = r_base["final_train_loss"]
    pl = r_prac["final_train_loss"]
    if bl and pl:
        print(f"  {'Final Train Loss':<30} {bl:>15.4f} {pl:>15.4f} {pl-bl:>+15.4f}")

    bel = r_base["final_eval_loss"]
    pel = r_prac["final_eval_loss"]
    if bel and pel:
        print(f"  {'Final Eval Loss':<30} {bel:>15.4f} {pel:>15.4f} {pel-bel:>+15.4f}")

    bs = r_base["samples_per_sec"]
    ps = r_prac["samples_per_sec"]
    tp_change = (ps / bs - 1) * 100 if bs > 0 else 0
    print(f"  {'Throughput (samples/sec)':<30} {bs:>15.2f} {ps:>15.2f} {tp_change:>+14.1f}%")

    bt = r_base["train_time_s"]
    pt = r_prac["train_time_s"]
    print(f"  {'Wall Time (sec)':<30} {bt:>15.1f} {pt:>15.1f} {pt-bt:>+15.1f}")

    if r_prac["prac_info"]:
        pi = r_prac["prac_info"]
        print(f"\n  PRAC details: {pi['n_layers']} layers, {pi['avg_savings_pct']:.1f}% avg activation savings")

    print("\n" + "=" * 80)

    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
