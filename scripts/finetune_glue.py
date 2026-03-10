#!/usr/bin/env python3
"""
RoBERTa GLUE fine-tuning script with PRAC activation compression.
"""

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats
from src.train import PRACTrainer

GLUE_TASKS = {
    "cola":  {"num_labels": 2, "metric": "matthews_correlation"},
    "sst2":  {"num_labels": 2, "metric": "accuracy"},
    "mrpc":  {"num_labels": 2, "metric": "f1"},
    "qqp":   {"num_labels": 2, "metric": "f1"},
    "stsb":  {"num_labels": 1, "metric": "pearson"},
    "mnli":  {"num_labels": 3, "metric": "accuracy"},
    "qnli":  {"num_labels": 2, "metric": "accuracy"},
    "rte":   {"num_labels": 2, "metric": "accuracy"},
    "wnli":  {"num_labels": 2, "metric": "accuracy"},
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
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    p = argparse.ArgumentParser(description="RoBERTa GLUE Fine-tuning with PRAC")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--task", type=str, required=True, choices=list(GLUE_TASKS))
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--use_prac", action="store_true")
    p.add_argument("--principal_rank", type=float, default=0.3)
    p.add_argument("--random_rank", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def compute_metrics(eval_pred, task_name):
    predictions, labels = eval_pred
    if task_name == "stsb":
        predictions = predictions.squeeze()
        pearson = float(np.corrcoef(predictions, labels)[0, 1])
        return {"pearson": pearson}

    predictions = np.argmax(predictions, axis=1)
    accuracy = float((predictions == labels).mean())

    if task_name in ("mrpc", "qqp"):
        from sklearn.metrics import f1_score
        f1 = float(f1_score(labels, predictions, average="binary"))
        return {"accuracy": accuracy, "f1": f1}
    if task_name == "cola":
        from sklearn.metrics import matthews_corrcoef
        mcc = float(matthews_corrcoef(labels, predictions))
        return {"accuracy": accuracy, "matthews_correlation": mcc}
    return {"accuracy": accuracy}


def preprocess(examples, tokenizer, task_name, max_length):
    k1, k2 = TASK_KEYS[task_name]
    if k2 is None:
        result = tokenizer(examples[k1], max_length=max_length, truncation=True)
    else:
        result = tokenizer(examples[k1], examples[k2],
                           max_length=max_length, truncation=True)
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"{args.model_name}-{args.task}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("glue", args.task)

    encoded = dataset.map(
        lambda x: preprocess(x, tokenizer, args.task, args.max_seq_length),
        batched=True, remove_columns=dataset["train"].column_names,
    )

    num_labels = GLUE_TASKS[args.task]["num_labels"]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels,
    )

    prac_config = None
    if args.use_prac:
        print("[INFO] Applying PRAC ...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank,
        )
        model = apply_prac_to_model(model, prac_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=GLUE_TASKS[args.task]["metric"],
        greater_is_better=True,
        report_to="tensorboard",
    )

    eval_split = "validation_matched" if args.task == "mnli" else "validation"

    trainer = PRACTrainer(
        prac_config=prac_config,
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_split],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, args.task),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("[INFO] Starting training ...")
    trainer.train()

    results = trainer.evaluate()
    print(f"[RESULT] {results}")
    trainer.save_model(os.path.join(output_dir, "final_model"))


if __name__ == "__main__":
    main()
