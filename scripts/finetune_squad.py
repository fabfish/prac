#!/usr/bin/env python3
"""
RoBERTa 在 SQuAD 上的微调脚本
支持 PRAC 激活压缩
"""

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import load_dataset
import argparse
import os
import sys
import numpy as np
from typing import Optional, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.prac import PRACConfig, apply_prac_to_model


def parse_args():
    parser = argparse.ArgumentParser(description="RoBERTa SQuAD Fine-tuning")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="roberta-base",
                       help="模型名称")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "squad_v2"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    
    # PRAC 参数
    parser.add_argument("--use_prac", action="store_true",
                       help="启用 PRAC 激活压缩")
    parser.add_argument("--principal_rank", type=float, default=0.3,
                       help="主子空间比例")
    parser.add_argument("--random_rank", type=float, default=0.3,
                       help="随机子空间比例")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=384,
                       help="最大序列长度")
    parser.add_argument("--doc_stride", type=int, default=128,
                       help="文档步长")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="warmup 比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 其他
    parser.add_argument("--bf16", action="store_true", help="使用 bfloat16")
    parser.add_argument("--fp16", action="store_true", help="使用 float16")
    
    return parser.parse_args()


def preprocess_training_examples(examples, tokenizer, max_length, doc_stride):
    """预处理训练数据"""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]
    
    # Tokenize
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    # 找到答案位置
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        
        # 如果没有答案，使用 CLS token
        if len(answer["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            # 找到 token 位置
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # 检查答案是否在当前 span 中
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # 找到答案的 token 位置
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    
    return tokenized


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f"{args.model_name}-{args.dataset}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 tokenizer 和模型
    print(f"🔄 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    
    # 应用 PRAC
    if args.use_prac:
        print("🚀 启用 PRAC 激活压缩...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank
        )
        model = apply_prac_to_model(model, prac_config)
    
    # 加载数据集
    print(f"🔄 加载 {args.dataset} 数据集")
    dataset = load_dataset(args.dataset)
    
    # 预处理
    print("🔄 预处理数据")
    train_dataset = dataset['train'].map(
        lambda x: preprocess_training_examples(x, tokenizer, args.max_seq_length, args.doc_stride),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        run_name=f"prac-{args.model_name}-{args.dataset}" if args.use_prac else f"baseline-{args.model_name}-{args.dataset}",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # 简化示例，实际应添加验证
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    # 训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 保存
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"✅ 模型已保存到 {output_dir}/final_model")


if __name__ == "__main__":
    main()
