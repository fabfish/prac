#!/usr/bin/env python3
"""
RoBERTa 在 GLUE 任务上的微调脚本
支持 PRAC 激活压缩
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset
import argparse
import os
import sys
import numpy as np
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.prac import PRACConfig, apply_prac_to_model


# GLUE 任务配置
GLUE_TASKS = {
    'cola': {'num_labels': 2, 'metric': 'matthews_correlation'},
    'sst2': {'num_labels': 2, 'metric': 'accuracy'},
    'mrpc': {'num_labels': 2, 'metric': 'f1'},
    'qqp': {'num_labels': 2, 'metric': 'f1'},
    'stsb': {'num_labels': 1, 'metric': 'pearson'},
    'mnli': {'num_labels': 3, 'metric': 'accuracy'},
    'qnli': {'num_labels': 2, 'metric': 'accuracy'},
    'rte': {'num_labels': 2, 'metric': 'accuracy'},
    'wnli': {'num_labels': 2, 'metric': 'accuracy'},
}


class PRACTrainer(Trainer):
    """集成 PRAC 的 Trainer"""
    
    def __init__(self, prac_config: Optional[PRACConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prac_config = prac_config
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """训练步骤，添加 PRAC 统计"""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 记录 PRAC 内存节省
        if self.prac_config and hasattr(model, 'get_prac_stats'):
            stats = model.get_prac_stats()
            if stats:
                total_savings = sum(s.get('savings_ratio', 0) for s in stats.values()) / len(stats)
                self.log({"prac_memory_savings": total_savings})
        
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="RoBERTa GLUE Fine-tuning")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="roberta-base",
                       help="模型名称 (roberta-base, roberta-large)")
    parser.add_argument("--task", type=str, required=True,
                       choices=list(GLUE_TASKS.keys()),
                       help="GLUE 任务名称")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    
    # PRAC 参数
    parser.add_argument("--use_prac", action="store_true",
                       help="启用 PRAC 激活压缩")
    parser.add_argument("--principal_rank", type=float, default=0.3,
                       help="主子空间比例 (0.0-1.0)")
    parser.add_argument("--random_rank", type=float, default=0.3,
                       help="随机子空间比例 (0.0-1.0)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 其他
    parser.add_argument("--bf16", action="store_true",
                       help="使用 bfloat16")
    parser.add_argument("--fp16", action="store_true",
                       help="使用 float16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="使用梯度检查点")
    
    return parser.parse_args()


def compute_metrics(eval_pred, task_name):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    if task_name == 'stsb':
        # 回归任务
        predictions = predictions.squeeze()
        pearson = np.corrcoef(predictions, labels)[0, 1]
        spearman = np.corrcoef(predictions.argsort().argsort(), 
                               labels.argsort().argsort())[0, 1]
        return {"pearson": pearson, "spearman": spearman}
    else:
        # 分类任务
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        
        if task_name in ['mrpc', 'qqp']:
            # 计算 F1
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, predictions, average='binary')
            return {"accuracy": accuracy, "f1": f1}
        elif task_name == 'cola':
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(labels, predictions)
            return {"accuracy": accuracy, "matthews_correlation": mcc}
        else:
            return {"accuracy": accuracy}


def preprocess_function(examples, tokenizer, task_name, max_length):
    """预处理数据"""
    task_to_keys = {
        'cola': ('sentence', None),
        'sst2': ('sentence', None),
        'mrpc': ('sentence1', 'sentence2'),
        'qqp': ('question1', 'question2'),
        'stsb': ('sentence1', 'sentence2'),
        'mnli': ('premise', 'hypothesis'),
        'qnli': ('question', 'sentence'),
        'rte': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }
    
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    if sentence2_key is None:
        # 单句任务
        texts = examples[sentence1_key]
        result = tokenizer(texts, max_length=max_length, truncation=True)
    else:
        # 双句任务
        texts = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*texts, max_length=max_length, truncation=True)
    
    if 'label' in examples:
        result['labels'] = examples['label']
    
    return result


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f"{args.model_name}-{args.task}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 tokenizer
    print(f"🔄 加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加载数据集
    print(f"🔄 加载 GLUE {args.task} 数据集")
    dataset = load_dataset("glue", args.task)
    
    # 预处理
    print("🔄 预处理数据")
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.task, args.max_seq_length),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 加载模型
    print(f"🔄 加载模型: {args.model_name}")
    num_labels = GLUE_TASKS[args.task]['num_labels']
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # 应用 PRAC
    if args.use_prac:
        print("🚀 启用 PRAC 激活压缩...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank
        )
        model = apply_prac_to_model(model, prac_config)
        print(f"   PRAC 配置: principal={args.principal_rank}, random={args.random_rank}")
    
    # 梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # 训练参数
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
        metric_for_best_model=GLUE_TASKS[args.task]['metric'],
        greater_is_better=True,
        report_to="tensorboard",
        run_name=f"prac-{args.model_name}-{args.task}" if args.use_prac else f"baseline-{args.model_name}-{args.task}",
    )
    
    # 创建 Trainer
    trainer = PRACTrainer(
        prac_config=PRACConfig(principal_rank=args.principal_rank, random_rank=args.random_rank) if args.use_prac else None,
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation_matched' if args.task == 'mnli' else 'validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, args.task),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 评估
    print("📝 最终评估...")
    eval_results = trainer.evaluate()
    print(f"   评估结果: {eval_results}")
    
    # 保存模型
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"✅ 模型已保存到 {output_dir}/final_model")
    
    # 打印 PRAC 统计
    if args.use_prac and hasattr(model, 'get_prac_stats'):
        stats = model.get_prac_stats()
        if stats:
            print("\n📊 PRAC 内存统计:")
            for name, s in list(stats.items())[:5]:  # 只显示前5层
                print(f"   {name}: {s.get('savings_ratio', 0):.1f}% 节省")


if __name__ == "__main__":
    main()
