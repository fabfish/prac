#!/usr/bin/env python3
"""
LLaMA instruction fine-tuning script with PRAC activation compression + LoRA.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import argparse
import os
import sys
import json
import copy
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats
from src.train import PRACTrainer


# ---------------------------------------------------------------------------
# Chat / instruction templates
# ---------------------------------------------------------------------------

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

ALPACA_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLaMA Instruction Fine-tuning with PRAC")

    # Model
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--dataset", type=str, default="alpaca",
                   choices=["alpaca", "dolly", "custom"])
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--chat_template", type=str, default="auto",
                   choices=["auto", "alpaca", "llama3"],
                   help="Prompt template. 'auto' picks llama3 for Llama-3.x models.")

    # PRAC
    p.add_argument("--use_prac", action="store_true")
    p.add_argument("--principal_rank", type=float, default=0.25)
    p.add_argument("--random_rank", type=float, default=0.25)
    p.add_argument("--prac_update_freq", type=int, default=200)

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Quantization
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap training samples (useful for smoke tests).")

    # Misc
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--deepspeed", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _pick_template(args):
    if args.chat_template == "auto":
        name_lower = args.model_name.lower()
        if "llama-3" in name_lower or "llama3" in name_lower:
            return "llama3"
        return "alpaca"
    return args.chat_template


def format_example(example, template_name):
    """Format a single Alpaca-style example into a full text string."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if template_name == "llama3":
        combined_inst = f"{instruction}\n{inp}" if inp else instruction
        return {"text": LLAMA3_TEMPLATE.format(instruction=combined_inst, output=output)}

    if inp:
        return {"text": ALPACA_INPUT_TEMPLATE.format(
            instruction=instruction, input=inp, output=output)}
    return {"text": ALPACA_TEMPLATE.format(instruction=instruction, output=output)}


def load_instruction_dataset(dataset_name, data_path, template_name, max_samples=None):
    if dataset_name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    elif dataset_name == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    elif dataset_name == "custom" and data_path:
        with open(data_path) as f:
            data = json.load(f)
        ds = Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    ds = ds.map(lambda ex: format_example(ex, template_name), remove_columns=ds.column_names)
    return ds


def tokenize_with_labels(examples, tokenizer, max_length):
    """
    Tokenize and build labels that mask the prompt portion with -100
    so only the completion contributes to the loss.
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for text in examples["text"]:
        # For Llama-3 template, the assistant header marks the split
        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            sep = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            parts = text.split(sep, 1)
            prompt_text = parts[0] + sep
        elif "### Response:\n" in text:
            sep = "### Response:\n"
            parts = text.split(sep, 1)
            prompt_text = parts[0] + sep
        else:
            prompt_text = ""

        tok_full = tokenizer(
            text, max_length=max_length, truncation=True,
            padding=False, return_attention_mask=False,
        )
        input_ids = tok_full["input_ids"]

        if prompt_text:
            tok_prompt = tokenizer(
                prompt_text, max_length=max_length, truncation=True,
                padding=False, return_attention_mask=False,
            )
            prompt_len = len(tok_prompt["input_ids"])
        else:
            prompt_len = 0

        labels = copy.deepcopy(input_ids)
        labels[:prompt_len] = [-100] * prompt_len

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append([1] * len(input_ids))

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_short = args.model_name.rstrip("/").split("/")[-1]
    output_dir = os.path.join(args.output_dir, f"{model_short}-prac-ft")
    os.makedirs(output_dir, exist_ok=True)

    template_name = _pick_template(args)
    print(f"[INFO] Using chat template: {template_name}")

    # ---- Quantization config ----
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ---- Load model & tokenizer ----
    print(f"[INFO] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else (
            torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # ---- Apply LoRA first (needs to find nn.Linear before PRAC wraps them) ----
    if args.use_lora:
        print(f"[INFO] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) ...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---- Apply PRAC (after LoRA -- wraps the inner nn.Linear / base_layer) ----
    prac_config = None
    if args.use_prac:
        print("[INFO] Applying PRAC activation compression ...")
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
            avg = sum(s["savings_pct"] for s in enabled) / len(enabled)
            print(f"[PRAC] {len(enabled)} layers, avg activation savings {avg:.1f}%")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ---- Load & tokenize dataset ----
    print(f"[INFO] Loading {args.dataset} dataset ...")
    ds = load_instruction_dataset(
        args.dataset, args.data_path, template_name,
        max_samples=args.max_train_samples,
    )
    n_total = len(ds)
    n_eval = min(200, max(1, n_total // 20))
    split = ds.train_test_split(test_size=n_eval, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[INFO] Tokenizing (max_seq_length={args.max_seq_length}) ...")
    train_ds = train_ds.map(
        lambda ex: tokenize_with_labels(ex, tokenizer, args.max_seq_length),
        batched=True, remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: tokenize_with_labels(ex, tokenizer, args.max_seq_length),
        batched=True, remove_columns=eval_ds.column_names,
    )

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        max_grad_norm=args.max_grad_norm,
        report_to="tensorboard",
        run_name=f"{'prac-' if args.use_prac else ''}{model_short}",
        deepspeed=args.deepspeed,
        seed=args.seed,
    )

    trainer = PRACTrainer(
        prac_config=prac_config,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, padding=True,
        ),
    )

    print("[INFO] Starting training ...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[INFO] Model saved to {final_dir}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
