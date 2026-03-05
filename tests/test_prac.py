"""
PRAC 单元测试
"""

import torch
import pytest
from src.prac import PRACConfig, PRACCompressor, PRACLinear


class TestPRACCompressor:
    """测试 PRAC 压缩器"""
    
    def test_initialization(self):
        """测试初始化"""
        config = PRACConfig(principal_rank=0.3, random_rank=0.3)
        compressor = PRACCompressor(config, feature_dim=768)
        
        assert compressor.r1 == int(768 * 0.3)
        assert compressor.r2 == int(768 * 0.3)
        assert compressor.k == (768 - compressor.r1) / compressor.r2
    
    def test_compression_decompression(self):
        """测试压缩和重建"""
        config = PRACConfig(principal_rank=0.3, random_rank=0.3)
        compressor = PRACCompressor(config, feature_dim=128)
        
        # 创建测试数据
        x = torch.randn(2, 64, 128)  # [batch, seq, dim]
        
        # 压缩
        XQ1, XQ2 = compressor.compress(x)
        
        # 检查尺寸
        assert XQ1.shape == (2, 64, compressor.r1)
        assert XQ2.shape == (2, 64, compressor.r2)
        
        # 重建
        x_recon = compressor.decompress(XQ1, XQ2)
        
        # 检查尺寸
        assert x_recon.shape == x.shape
        
        # 检查重建误差不太大
        error = torch.norm(x - x_recon) / torch.norm(x)
        assert error < 0.5  # 允许一定误差
    
    def test_memory_savings(self):
        """测试内存节省计算"""
        config = PRACConfig(principal_rank=0.3, random_rank=0.3)
        compressor = PRACCompressor(config, feature_dim=100)
        
        savings = compressor.memory_savings()
        expected = 1.0 - (compressor.r1 + compressor.r2) / 100
        
        assert abs(savings - expected) < 1e-6
    
    def test_subspace_update(self):
        """测试子空间更新"""
        config = PRACConfig(
            principal_rank=0.3,
            random_rank=0.3,
            principal_update_freq=10
        )
        compressor = PRACCompressor(config, feature_dim=64)
        
        x = torch.randn(4, 32, 64)
        
        # 初始状态
        assert compressor.Q1 is None
        assert compressor.Q2 is None
        
        # 压缩触发更新
        compressor.compress(x)
        
        # 检查子空间已创建
        assert compressor.Q1 is not None
        assert compressor.Q2 is not None
        assert compressor.Q1.shape == (64, compressor.r1)
        assert compressor.Q2.shape == (64, compressor.r2)


class TestPRACLinear:
    """测试 PRAC 线性层"""
    
    def test_forward_pass(self):
        """测试前向传播"""
        config = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(128, 256, config)
        
        x = torch.randn(2, 16, 128)
        
        # 前向传播
        output = layer(x)
        
        # 检查输出尺寸
        assert output.shape == (2, 16, 256)
    
    def test_memory_stats(self):
        """测试内存统计"""
        config = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(100, 200, config)
        
        stats = layer.get_memory_stats()
        
        assert 'original_memory' in stats
        assert 'compressed_memory' in stats
        assert 'savings_ratio' in stats
        assert stats['enabled'] is True
    
    def test_no_compression_for_small_dim(self):
        """测试小维度不压缩"""
        config = PRACConfig(min_dim_threshold=100)
        layer = PRACLinear(64, 128, config)  # 64 < 100
        
        assert layer.compressor is None
        
        stats = layer.get_memory_stats()
        assert stats['enabled'] is False


class TestPRACConfig:
    """测试 PRAC 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PRACConfig()
        
        assert config.principal_rank == 0.3
        assert config.random_rank == 0.3
        assert config.principal_update_freq == 100
        assert config.random_update_freq == 50
        assert config.share_subspace is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PRACConfig(
            principal_rank=0.4,
            random_rank=0.2,
            principal_update_freq=200
        )
        
        assert config.principal_rank == 0.4
        assert config.random_rank == 0.2
        assert config.principal_update_freq == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
