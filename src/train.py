"""
PRAC 训练脚本
用于预训练或微调 LLM 时使用 PRAC 激活压缩
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import argparse
import os
import sys
from typing import Optional

# 添加 src 到路径
sys.path.append(os.path.dirname(__file__))
from prac import PRACConfig, apply_prac_to_model


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
            total_savings = sum(s['savings_ratio'] for s in stats.values()) / len(stats)
            self.log({"prac_memory_savings": total_savings})
        
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="PRAC 训练脚本")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="预训练模型名称")
    parser.add_argument("--output_dir", type=str, default="./prac_output",
                       help="输出目录")
    
    # PRAC 参数
    parser.add_argument("--use_prac", action="store_true",
                       help="启用 PRAC 激活压缩")
    parser.add_argument("--principal_rank", type=float, default=0.3,
                       help="主子空间比例 (0.0-1.0)")
    parser.add_argument("--random_rank", type=float, default=0.3,
                       help="随机子空间比例 (0.0-1.0)")
    parser.add_argument("--principal_update_freq", type=int, default=100,
                       help="主子空间更新频率")
    parser.add_argument("--random_update_freq", type=int, default=50,
                       help="随机子空间更新频率")
    parser.add_argument("--share_subspace", action="store_true", default=True,
                       help="启用子空间共享")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    
    # LoRA 参数 (可选)
    parser.add_argument("--use_lora", action="store_true",
                       help="使用 LoRA 微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--bf16", action="store_true",
                       help="使用 bfloat16")
    parser.add_argument("--fp16", action="store_true",
                       help="使用 float16")
    
    return parser.parse_args()


def setup_model_and_tokenizer(args):
    """设置模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto"
    )
    
    # 应用 PRAC
    if args.use_prac:
        print("🚀 启用 PRAC 激活压缩...")
        prac_config = PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank,
            principal_update_freq=args.principal_update_freq,
            random_update_freq=args.random_update_freq,
            share_subspace=args.share_subspace
        )
        model = apply_prac_to_model(model, prac_config)
        
        # 打印内存节省统计
        total_params = 0
        compressed_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'get_memory_stats'):
                stats = module.get_memory_stats()
                if stats['enabled']:
                    total_params += stats['original_memory']
                    compressed_params += stats['compressed_memory']
        
        if total_params > 0:
            savings = (1 - compressed_params / total_params) * 100
            print(f"📊 PRAC 内存节省: {savings:.1f}%")
    
    # 应用 LoRA (可选)
    if args.use_lora:
        print("🎯 启用 LoRA 微调...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置模型
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        run_name=f"prac_{args.model_name}" if args.use_prac else f"baseline_{args.model_name}",
    )
    
    # 创建 Trainer
    trainer = PRACTrainer(
        prac_config=PRACConfig(
            principal_rank=args.principal_rank,
            random_rank=args.random_rank
        ) if args.use_prac else None,
        model=model,
        args=training_args,
        # 这里应该添加数据集，简化示例省略
        train_dataset=None,  # 用户需要填充
        eval_dataset=None,   # 用户需要填充
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"✅ 模型已保存到 {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
