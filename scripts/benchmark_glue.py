#!/usr/bin/env python3
"""
GLUE benchmark: Baseline vs PRAC comparison.

Runs RoBERTa-Base on selected GLUE tasks with and without PRAC,
reports accuracy/F1/MCC and peak GPU memory.

Paper reference (Section 5.2):
  - Model: RoBERTa-Base, Batch size: 16, Epochs: 30
  - PRAC rank: r1+r2, interval=200 steps
  - Tasks: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE
"""

import torch
import numpy as np
import gc
import json
import time
import os
import sys
import argparse

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats

# ── GLUE task definitions ──────────────────────────────────────────────────

GLUE_TASKS = {
    "cola":  {"num_labels": 2, "metric": "matthews_correlation"},
    "sst2":  {"num_labels": 2, "metric": "accuracy"},
    "mrpc":  {"num_labels": 2, "metric": "f1"},
    "qqp":   {"num_labels": 2, "metric": "f1"},
    "stsb":  {"num_labels": 1, "metric": "pearson"},
    "mnli":  {"num_labels": 3, "metric": "accuracy"},
    "qnli":  {"num_labels": 2, "metric": "accuracy"},
    "rte":   {"num_labels": 2, "metric": "accuracy"},
}

TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
}


def compute_metrics(eval_pred, task_name):
    predictions, labels = eval_pred
    if task_name == "stsb":
        predictions = predictions.squeeze()
        return {"pearson": float(np.corrcoef(predictions, labels)[0, 1])}
    predictions = np.argmax(predictions, axis=1)
    acc = float((predictions == labels).mean())
    if task_name in ("mrpc", "qqp"):
        from sklearn.metrics import f1_score
        return {"accuracy": acc, "f1": float(f1_score(labels, predictions, average="binary"))}
    if task_name == "cola":
        from sklearn.metrics import matthews_corrcoef
        return {"accuracy": acc, "matthews_correlation": float(matthews_corrcoef(labels, predictions))}
    return {"accuracy": acc}


def preprocess(examples, tokenizer, task_name, max_length):
    k1, k2 = TASK_KEYS[task_name]
    if k2 is None:
        result = tokenizer(examples[k1], max_length=max_length, truncation=True)
    else:
        result = tokenizer(examples[k1], examples[k2], max_length=max_length, truncation=True)
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_single_experiment(task_name, use_prac, args):
    """Run a single GLUE experiment, return results dict."""
    clear_gpu()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = "prac" if use_prac else "baseline"
    print(f"\n{'='*60}")
    print(f"  Task: {task_name} | Mode: {tag}")
    print(f"{'='*60}")

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load & preprocess data
    dataset = load_dataset("glue", task_name)
    encoded = dataset.map(
        lambda x: preprocess(x, tokenizer, task_name, args.max_seq_length),
        batched=True, remove_columns=dataset["train"].column_names,
    )

    # Load model on GPU
    num_labels = GLUE_TASKS[task_name]["num_labels"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).cuda()

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
            print(f"  [PRAC] {len(enabled)} layers, avg savings {avg_sav:.1f}%")

    out_dir = os.path.join(args.output_dir, f"{task_name}-{tag}")
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=GLUE_TASKS[task_name]["metric"],
        greater_is_better=(task_name != "stsb" or True),
        report_to="none",
        seed=args.seed,
    )

    eval_split = "validation_matched" if task_name == "mnli" else "validation"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_split],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, task_name),
    )

    # Record memory before training
    clear_gpu()
    mem_before = torch.cuda.memory_allocated(0) / 1024**2

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Peak memory during training
    peak_mem = torch.cuda.max_memory_allocated(0) / 1024**2

    # Evaluate
    eval_results = trainer.evaluate()
    primary_metric = GLUE_TASKS[task_name]["metric"]
    primary_value = eval_results.get(f"eval_{primary_metric}", None)

    result = {
        "task": task_name,
        "mode": tag,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "eval_results": {k: v for k, v in eval_results.items() if isinstance(v, (int, float))},
        "peak_gpu_mem_mb": peak_mem,
        "train_time_s": train_time,
        "prac_info": prac_info,
    }

    print(f"  Result: {primary_metric} = {primary_value:.4f}")
    print(f"  Peak GPU memory: {peak_mem:.0f} MB")
    print(f"  Train time: {train_time:.1f}s")

    # Cleanup
    del model, trainer
    clear_gpu()

    return result


def main():
    p = argparse.ArgumentParser(description="GLUE Benchmark: Baseline vs PRAC")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["sst2", "mrpc", "cola"],
                   choices=list(GLUE_TASKS))
    p.add_argument("--output_dir", type=str, default="./glue_benchmark")
    p.add_argument("--principal_rank", type=float, default=0.3)
    p.add_argument("--random_rank", type=float, default=0.3)
    p.add_argument("--prac_update_freq", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  GLUE Benchmark: Baseline vs PRAC")
    print(f"  Model: {args.model_name}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Epochs: {args.num_epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print(f"  PRAC: r1={args.principal_rank}, r2={args.random_rank}, freq={args.prac_update_freq}")
    print(f"  GPU: {args.gpu}")
    print("=" * 60)

    all_results = []

    for task in args.tasks:
        # Baseline
        r_base = run_single_experiment(task, use_prac=False, args=args)
        all_results.append(r_base)

        # PRAC
        r_prac = run_single_experiment(task, use_prac=True, args=args)
        all_results.append(r_prac)

    # ── Summary table ──
    print("\n\n" + "=" * 80)
    print("  SUMMARY: Baseline vs PRAC")
    print("=" * 80)
    header = f"{'Task':<8} {'Metric':<22} {'Baseline':>10} {'PRAC':>10} {'Delta':>10} {'Mem Base':>10} {'Mem PRAC':>10} {'Mem Save':>10}"
    print(header)
    print("-" * 80)

    for task in args.tasks:
        base = next(r for r in all_results if r["task"] == task and r["mode"] == "baseline")
        prac = next(r for r in all_results if r["task"] == task and r["mode"] == "prac")
        metric = base["primary_metric"]
        bv = base["primary_value"]
        pv = prac["primary_value"]
        delta = pv - bv if (bv is not None and pv is not None) else float("nan")
        bm = base["peak_gpu_mem_mb"]
        pm = prac["peak_gpu_mem_mb"]
        mem_save = (1 - pm / bm) * 100 if bm > 0 else 0

        print(f"{task:<8} {metric:<22} {bv:>10.4f} {pv:>10.4f} {delta:>+10.4f} {bm:>9.0f}M {pm:>9.0f}M {mem_save:>9.1f}%")

    print("=" * 80)

    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
