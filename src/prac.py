"""
PRAC: Principal-Random Subspace for LLM Activation Compression
核心实现模块

论文: PRAC: Principal-Random Subspace for LLM Activation Compression 
      and Memory-Efficient Training
作者: Yanyi Li, Yimu Zhang, Cong Fang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from dataclasses import dataclass


@dataclass
class PRACConfig:
    """PRAC 配置参数"""
    # 主子空间秩 (保留的主要信息维度)
    principal_rank: int = 0.3  # 默认保留 30% 维度
    
    # 随机子空间秩 (用于近似尾部信息)
    random_rank: int = 0.3  # 默认 30% 维度
    
    # 缩放因子 (理论最优值 k = (n-r1)/r2)
    scaling_factor: Optional[float] = None
    
    # 子空间更新频率 (惰性更新策略)
    principal_update_freq: int = 100  # 每100步更新主子空间
    random_update_freq: int = 50      # 每50步更新随机子空间
    
    # 是否使用子空间共享 (跨层共享投影矩阵)
    share_subspace: bool = True
    
    # 分层配置 (不同层使用不同秩)
    layer_configs: Optional[Dict[str, Tuple[float, float]]] = None
    
    # 最小维度阈值 (低于此值不压缩)
    min_dim_threshold: int = 64


class PRACCompressor:
    """
    PRAC 激活值压缩器
    
    核心算法：
    1. 对激活值 X 进行 SVD 分解
    2. 提取主子空间 Q1 (前 r1 个右奇异向量)
    3. 从正交补空间随机采样 Q2
    4. 压缩存储：XQ1 和 XQ2
    5. 重建：X̃ = (XQ1)Q1ᵀ + k(XQ2)Q2ᵀ
    """
    
    def __init__(self, config: PRACConfig, feature_dim: int):
        self.config = config
        self.feature_dim = feature_dim
        
        # 计算实际秩
        self.r1 = max(1, int(feature_dim * config.principal_rank))
        self.r2 = max(1, int(feature_dim * config.random_rank))
        
        # 理论最优缩放因子 k = (n-r1)/r2
        if config.scaling_factor is None:
            self.k = (feature_dim - self.r1) / self.r2
        else:
            self.k = config.scaling_factor
        
        # 子空间投影矩阵
        self.Q1: Optional[torch.Tensor] = None  # 主子空间 [n, r1]
        self.Q2: Optional[torch.Tensor] = None  # 随机子空间 [n, r2]
        
        # 更新计数器
        self.step_count = 0
        
    def update_subspaces(self, activations: torch.Tensor):
        """
        更新子空间投影矩阵
        
        Args:
            activations: [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        """
        # 惰性更新策略
        if (self.step_count % self.config.principal_update_freq == 0 or 
            self.Q1 is None):
            self._update_principal_subspace(activations)
            
        if (self.step_count % self.config.random_update_freq == 0 or 
            self.Q2 is None):
            self._update_random_subspace()
            
        self.step_count += 1
    
    def _update_principal_subspace(self, activations: torch.Tensor):
        """通过 SVD 更新主子空间"""
        # 合并 batch 和 seq 维度
        if activations.dim() == 3:
            X = activations.reshape(-1, activations.shape[-1])  # [B*S, n]
        else:
            X = activations
            
        # SVD 分解: X = U Σ V^T
        # 我们只需要右奇异向量 V 的前 r1 列
        try:
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            # Vh 是 V^T，取前 r1 行后转置得到 V[:, :r1]
            self.Q1 = Vh[:self.r1, :].T.contiguous()  # [n, r1]
        except:
            # 如果 SVD 失败，使用随机初始化
            self.Q1 = torch.randn(self.feature_dim, self.r1, device=X.device)
            self.Q1, _ = torch.linalg.qr(self.Q1)
    
    def _update_random_subspace(self):
        """从正交补空间随机采样"""
        if self.Q1 is None:
            # 如果主子空间未初始化，随机初始化 Q2
            self.Q2 = torch.randn(self.feature_dim, self.r2)
            self.Q2, _ = torch.linalg.qr(self.Q2)
        else:
            # 从 Q1 的正交补空间采样
            # 方法：生成随机矩阵，投影到 Q1 的零空间
            device = self.Q1.device
            Z = torch.randn(self.feature_dim, self.r2, device=device)
            
            # Gram-Schmidt 正交化，确保与 Q1 正交
            # Z = Z - Q1(Q1^T Z)
            Z = Z - self.Q1 @ (self.Q1.T @ Z)
            
            # QR 分解得到标准正交基
            self.Q2, _ = torch.linalg.qr(Z)
    
    def compress(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩激活值
        
        Args:
            activations: [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
            
        Returns:
            XQ1: 主子空间投影 [batch_size, seq_len, r1]
            XQ2: 随机子空间投影 [batch_size, seq_len, r2]
        """
        if self.Q1 is None or self.Q2 is None:
            self.update_subspaces(activations)
            
        # 确保在同一设备
        device = activations.device
        if self.Q1.device != device:
            self.Q1 = self.Q1.to(device)
            self.Q2 = self.Q2.to(device)
        
        # 压缩投影
        XQ1 = activations @ self.Q1  # [B, S, r1]
        XQ2 = activations @ self.Q2  # [B, S, r2]
        
        return XQ1, XQ2
    
    def decompress(self, XQ1: torch.Tensor, XQ2: torch.Tensor) -> torch.Tensor:
        """
        重建激活值
        
        公式: X̃ = (XQ1)Q1ᵀ + k(XQ2)Q2ᵀ
        
        Args:
            XQ1: 主子空间投影 [batch_size, seq_len, r1]
            XQ2: 随机子空间投影 [batch_size, seq_len, r2]
            
        Returns:
            X_reconstructed: 重建的激活值 [batch_size, seq_len, feature_dim]
        """
        device = XQ1.device
        if self.Q1.device != device:
            self.Q1 = self.Q1.to(device)
            self.Q2 = self.Q2.to(device)
        
        # 主子空间重建
        X_principal = XQ1 @ self.Q1.T  # [B, S, n]
        
        # 随机子空间重建 (带缩放因子)
        X_random = self.k * (XQ2 @ self.Q2.T)  # [B, S, n]
        
        # 合并
        return X_principal + X_random
    
    def memory_savings(self) -> float:
        """计算内存节省比例"""
        original = self.feature_dim
        compressed = self.r1 + self.r2
        return 1.0 - (compressed / original)


class PRACLinear(nn.Module):
    """
    支持 PRAC 激活压缩的线性层
    
    用于替换 Transformer 中的 MLP 和 Attention 投影层
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 config: PRACConfig, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 原始线性层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # PRAC 压缩器 (只在输入维度较大时启用)
        self.compressor = None
        if in_features >= config.min_dim_threshold:
            self.compressor = PRACCompressor(config, in_features)
        
        # 存储压缩后的激活值 (用于反向传播)
        self.compressed_activations = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，支持激活值压缩
        
        Args:
            x: 输入 [batch_size, seq_len, in_features]
            
        Returns:
            output: [batch_size, seq_len, out_features]
        """
        # 如果启用了压缩
        if self.compressor is not None and self.training:
            # 更新子空间
            self.compressor.update_subspaces(x)
            
            # 压缩并存储 (用于反向传播时重建)
            XQ1, XQ2 = self.compressor.compress(x)
            self.compressed_activations = (XQ1, XQ2)
            
            # 重建 (有微小误差，但可接受)
            x_reconstructed = self.compressor.decompress(XQ1, XQ2)
            
            # 使用重建的激活值进行线性变换
            return self.linear(x_reconstructed)
        else:
            # 不压缩，直接前向传播
            return self.linear(x)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计信息"""
        if self.compressor is None:
            return {
                "original_memory": self.in_features,
                "compressed_memory": self.in_features,
                "savings_ratio": 0.0,
                "enabled": False
            }
        
        savings = self.compressor.memory_savings()
        return {
            "original_memory": self.in_features,
            "compressed_memory": self.compressor.r1 + self.compressor.r2,
            "savings_ratio": savings * 100,  # 百分比
            "enabled": True,
            "principal_rank": self.compressor.r1,
            "random_rank": self.compressor.r2
        }


class PRACTransformerBlock(nn.Module):
    """
    集成 PRAC 的 Transformer Block
    """
    
    def __init__(self, hidden_size: int, num_heads: int, 
                 intermediate_size: int, config: PRACConfig):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention 投影层 (使用 PRAC)
        self.q_proj = PRACLinear(hidden_size, hidden_size, config)
        self.k_proj = PRACLinear(hidden_size, hidden_size, config)
        self.v_proj = PRACLinear(hidden_size, hidden_size, config)
        self.o_proj = PRACLinear(hidden_size, hidden_size, config)
        
        # MLP 层 (使用 PRAC)
        self.gate_proj = PRACLinear(hidden_size, intermediate_size, config)
        self.up_proj = PRACLinear(hidden_size, intermediate_size, config)
        self.down_proj = PRACLinear(intermediate_size, hidden_size, config)
        
        # LayerNorm
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None):
        """前向传播"""
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 简化的 attention (实际应使用 flash attention 等)
        attn_output = self._attention(q, k, v, attention_mask)
        attn_output = self.o_proj(attn_output)
        
        hidden_states = residual + attn_output
        residual = hidden_states
        
        # MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        mlp_output = self.down_proj(gate * up)
        
        hidden_states = residual + mlp_output
        
        return hidden_states
    
    def _attention(self, q, k, v, mask=None):
        """简化的 scaled dot-product attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    def get_prac_stats(self):
        """获取所有 PRAC 层的统计信息"""
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, PRACLinear):
                stats[name] = module.get_memory_stats()
        return stats


def apply_prac_to_model(model: nn.Module, config: PRACConfig):
    """
    将 PRAC 应用到现有模型的所有线性层
    
    Args:
        model: 原始模型
        config: PRAC 配置
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 替换为 PRACLinear
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            prac_linear = PRACLinear(
                module.in_features,
                module.out_features,
                config,
                bias=module.bias is not None
            )
            # 复制权重
            prac_linear.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                prac_linear.linear.bias.data = module.bias.data.clone()
            
            setattr(parent, child_name, prac_linear)
    
    return model


if __name__ == "__main__":
    # 简单测试
    config = PRACConfig(principal_rank=0.3, random_rank=0.3)
    compressor = PRACCompressor(config, feature_dim=768)
    
    # 模拟激活值
    x = torch.randn(2, 128, 768)  # [batch, seq, dim]
    
    # 压缩
    XQ1, XQ2 = compressor.compress(x)
    print(f"原始尺寸: {x.shape}, 内存: {x.numel()}")
    print(f"压缩后: XQ1 {XQ1.shape}, XQ2 {XQ2.shape}, 总内存: {XQ1.numel() + XQ2.numel()}")
    print(f"内存节省: {compressor.memory_savings() * 100:.1f}%")
    
    # 重建
    x_recon = compressor.decompress(XQ1, XQ2)
    reconstruction_error = torch.norm(x - x_recon) / torch.norm(x)
    print(f"重建误差: {reconstruction_error.item():.4f}")
