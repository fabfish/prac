#!/usr/bin/env python3
"""
RoBERTa SQuAD fine-tuning script with PRAC activation compression.
"""

import torch
import numpy as np
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import load_dataset
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, apply_prac_to_model


def parse_args():
    p = argparse.ArgumentParser(description="RoBERTa SQuAD Fine-tuning with PRAC")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--dataset", type=str, default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--use_prac", action="store_true")
    p.add_argument("--principal_rank", type=float, default=0.3)
    p.add_argument("--random_rank", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def preprocess_training(examples, tokenizer, max_length, doc_stride):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions, contexts,
        max_length=max_length, truncation="only_second",
        stride=doc_stride, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_map = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_map):
        input_ids = tokenized["input_ids"][i]
        cls_idx = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tokenized.sequence_ids(i)
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        if not answer["answer_start"]:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        tok_start = 0
        while seq_ids[tok_start] != 1:
            tok_start += 1
        tok_end = len(input_ids) - 1
        while seq_ids[tok_end] != 1:
            tok_end -= 1

        if not (offsets[tok_start][0] <= start_char and offsets[tok_end][1] >= end_char):
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
        else:
            while tok_start < len(offsets) and offsets[tok_start][0] <= start_char:
                tok_start += 1
            start_positions.append(tok_start - 1)
            while offsets[tok_end][1] >= end_char:
                tok_end -= 1
            end_positions.append(tok_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"{args.model_name}-{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    if args.use_prac:
        print("[INFO] Applying PRAC ...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank, random_rank=args.random_rank,
        )
        model = apply_prac_to_model(model, prac_config)

    dataset = load_dataset(args.dataset)
    train_ds = dataset["train"].map(
        lambda x: preprocess_training(x, tokenizer, args.max_seq_length, args.doc_stride),
        batched=True, remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16, fp16=args.fp16,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    print("[INFO] Starting training ...")
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"[INFO] Model saved to {output_dir}/final_model")


if __name__ == "__main__":
    main()
