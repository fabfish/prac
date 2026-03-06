"""
LLaMA 模型适配器
支持 PRAC 激活压缩
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from src.prac import PRACConfig, apply_prac_to_model


def load_llama_with_prac(
    model_name: str,
    prac_config: Optional[PRACConfig] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    **kwargs
):
    """
    加载 LLaMA 模型，可选启用 PRAC
    
    Args:
        model_name: 模型名称 (如 "meta-llama/Llama-2-7b-hf")
        prac_config: PRAC 配置，None 则不启用
        load_in_4bit: 是否使用 4-bit 量化
        load_in_8bit: 是否使用 8-bit 量化
        torch_dtype: 数据类型
        device_map: 设备映射策略
        **kwargs: 其他 transformers 参数
        
    Returns:
        加载好的模型
    """
    from transformers import BitsAndBytesConfig
    
    # 量化配置
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        **kwargs
    )
    
    # 应用 PRAC
    if prac_config is not None:
        print(f"🚀 为 LLaMA 启用 PRAC 激活压缩...")
        print(f"   模型: {model_name}")
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
            print(f"📊 LLaMA + PRAC 内存节省: {savings:.1f}%")
    
    return model


def load_llama_tokenizer(model_name: str, **kwargs):
    """
    加载 LLaMA Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        **kwargs
    )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    return tokenizer


class LLaMAWithPRAC(nn.Module):
    """
    包装器：LLaMA + PRAC
    """
    
    def __init__(
        self,
        model_name: str,
        prac_enabled: bool = True,
        principal_rank: float = 0.25,
        random_rank: float = 0.25,
        **load_kwargs
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
        self.model = load_llama_with_prac(
            model_name,
            prac_config=prac_config,
            **load_kwargs
        )
        self.prac_enabled = prac_enabled
        self.model_name = model_name
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """生成文本"""
        return self.model.generate(*args, **kwargs)
    
    def get_prac_stats(self):
        """获取 PRAC 统计信息"""
        if not self.prac_enabled:
            return None
        
        stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_memory_stats'):
                stats[name] = module.get_memory_stats()
        return stats
    
    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点"""
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def enable_input_require_grads(self):
        """启用输入梯度"""
        self.model.enable_input_require_grads()
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """从本地加载模型"""
        return cls(model_path, **kwargs)


# 支持的 LLaMA 模型列表
SUPPORTED_LLAMA_MODELS = [
    # LLaMA 1
    "meta-llama/Llama-1-7b",
    "meta-llama/Llama-1-13b",
    "meta-llama/Llama-1-30b",
    "meta-llama/Llama-1-65b",
    # LLaMA 2
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    # LLaMA 3
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    # 中文 LLaMA
    "hfl/chinese-llama-2-7b",
    "hfl/chinese-alpaca-2-7b",
]
