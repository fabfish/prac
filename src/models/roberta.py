"""
RoBERTa 模型适配器
支持 PRAC 激活压缩
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from typing import Optional
from src.prac import PRACConfig, apply_prac_to_model


def load_roberta_with_prac(
    model_name: str,
    num_labels: int = 2,
    prac_config: Optional[PRACConfig] = None,
    **kwargs
):
    """
    加载 RoBERTa 模型，可选启用 PRAC
    
    Args:
        model_name: 模型名称 (如 "roberta-base", "roberta-large")
        num_labels: 分类标签数
        prac_config: PRAC 配置，None 则不启用
        **kwargs: 其他 transformers 参数
        
    Returns:
        加载好的模型
    """
    # 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **kwargs
    )
    
    # 应用 PRAC
    if prac_config is not None:
        print(f"🚀 为 RoBERTa 启用 PRAC 激活压缩...")
        model = apply_prac_to_model(model, prac_config)
        
        # 计算并打印内存节省
        total_params = 0
        compressed_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'get_memory_stats'):
                stats = module.get_memory_stats()
                if stats.get('enabled'):
                    total_params += stats['original_memory']
                    compressed_params += stats['compressed_memory']
        
        if total_params > 0:
            savings = (1 - compressed_params / total_params) * 100
            print(f"📊 RoBERTa + PRAC 内存节省: {savings:.1f}%")
    
    return model


class RoBERTaWithPRAC(nn.Module):
    """
    包装器：RoBERTa + PRAC
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        prac_enabled: bool = True,
        principal_rank: float = 0.3,
        random_rank: float = 0.3,
    ):
        super().__init__()
        
        # PRAC 配置
        prac_config = None
        if prac_enabled:
            prac_config = PRACConfig(
                principal_rank=principal_rank,
                random_rank=random_rank
            )
        
        # 加载模型
        self.model = load_roberta_with_prac(
            model_name,
            num_labels=num_labels,
            prac_config=prac_config
        )
        self.prac_enabled = prac_enabled
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_prac_stats(self):
        """获取 PRAC 统计信息"""
        if not self.prac_enabled:
            return None
        
        stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_memory_stats'):
                stats[name] = module.get_memory_stats()
        return stats
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """从本地加载模型"""
        # 简化实现，实际应读取配置文件
        return cls(model_path, **kwargs)
