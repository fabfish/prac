#!/usr/bin/env python3
"""
LLaMA 指令微调脚本
支持 PRAC 激活压缩 + LoRA
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse
import os
import sys
import json
from typing import Optional, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.prac import PRACConfig, apply_prac_to_model


# 指令数据集模板
INSTRUCTION_TEMPLATES = {
    "alpaca": {
        "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "with_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    },
    "chatml": {
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{input}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{output}<|im_end|>\n",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA Instruction Fine-tuning")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, required=True,
                       help="模型名称 (如 meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--dataset", type=str, default="alpaca",
                       choices=["alpaca", "dolly", "sharegpt", "custom"],
                       help="数据集名称")
    parser.add_argument("--data_path", type=str, default=None,
                       help="自定义数据路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    
    # PRAC 参数
    parser.add_argument("--use_prac", action="store_true",
                       help="启用 PRAC 激活压缩")
    parser.add_argument("--principal_rank", type=float, default=0.25,
                       help="主子空间比例")
    parser.add_argument("--random_rank", type=float, default=0.25,
                       help="随机子空间比例")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true",
                       help="使用 LoRA 微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # 量化参数
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="4-bit 量化加载")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="8-bit 量化加载")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=0.3,
                       help="梯度裁剪")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 其他
    parser.add_argument("--bf16", action="store_true", help="使用 bfloat16")
    parser.add_argument("--fp16", action="store_true", help="使用 float16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="使用梯度检查点")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed 配置文件路径")
    parser.add_argument("--group_by_length", action="store_true",
                       help="按长度分组")
    
    return parser.parse_args()


def format_alpaca_example(example):
    """格式化 Alpaca 数据"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    if input_text:
        prompt = INSTRUCTION_TEMPLATES["alpaca"]["with_input"].format(
            instruction=instruction,
            input=input_text
        )
    else:
        prompt = INSTRUCTION_TEMPLATES["alpaca"]["prompt"].format(
            instruction=instruction
        )
    
    text = prompt + output
    return {"text": text, "prompt": prompt, "completion": output}


def format_dolly_example(example):
    """格式化 Dolly 数据"""
    instruction = example.get('instruction', '')
    context = example.get('context', '')
    response = example.get('response', '')
    
    if context:
        text = f"Instruction: {instruction}\nContext: {context}\nResponse: {response}"
    else:
        text = f"Instruction: {instruction}\nResponse: {response}"
    
    return {"text": text}


def load_instruction_dataset(dataset_name, data_path=None):
    """加载指令微调数据集"""
    if dataset_name == "alpaca":
        # 使用 tatsu-lab/alpaca 数据集
        dataset = load_dataset("tatsu-lab/alpaca")
        dataset = dataset.map(format_alpaca_example)
        return dataset['train'], dataset['train'].select(range(100))  # 简化：用部分数据做验证
    
    elif dataset_name == "dolly":
        dataset = load_dataset("databricks/databricks-dolly-15k")
        dataset = dataset.map(format_dolly_example)
        return dataset['train'], dataset['train'].select(range(100))
    
    elif dataset_name == "custom" and data_path:
        # 加载自定义 JSON 数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        # 转换为 Dataset
        from datasets import Dataset
        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_alpaca_example)
        return dataset, dataset.select(range(min(100, len(dataset))))
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def preprocess_function(examples, tokenizer, max_length):
    """预处理数据"""
    texts = examples["text"]
    
    # Tokenize
    result = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    
    # 对于因果语言模型，labels = input_ids
    result["labels"] = result["input_ids"].copy()
    
    return result


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    model_name_short = args.model_name.split('/')[-1]
    output_dir = os.path.join(args.output_dir, f"{model_name_short}-{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 量化配置
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
    
    # 加载模型和 tokenizer
    print(f"🔄 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 准备模型用于训练（量化时）
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # 应用 PRAC
    if args.use_prac:
        print("🚀 启用 PRAC 激活压缩...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank
        )
        model = apply_prac_to_model(model, prac_config)
        print(f"   PRAC 配置: principal={args.principal_rank}, random={args.random_rank}")
    
    # 应用 LoRA
    if args.use_lora:
        print(f"🎯 启用 LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
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
    
    # 梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # 加载数据集
    print(f"🔄 加载 {args.dataset} 数据集")
    train_dataset, eval_dataset = load_instruction_dataset(args.dataset, args.data_path)
    
    # 预处理
    print("🔄 预处理数据")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # 训练参数
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
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        max_grad_norm=args.max_grad_norm,
        group_by_length=args.group_by_length,
        report_to="tensorboard",
        run_name=f"prac-{model_name_short}-{args.dataset}" if args.use_prac else f"baseline-{model_name_short}-{args.dataset}",
        deepspeed=args.deepspeed,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    
    # 训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"✅ 模型已保存到 {output_dir}/final_model")
    
    # 保存配置
    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "use_prac": args.use_prac,
        "principal_rank": args.principal_rank if args.use_prac else None,
        "random_rank": args.random_rank if args.use_prac else None,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
