"""Unit tests for the PRAC core module."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prac import PRACConfig, PRACCompressor, PRACLinear, PRACFunction


class TestPRACConfig:
    def test_defaults(self):
        cfg = PRACConfig()
        assert cfg.principal_rank == 0.3
        assert cfg.random_rank == 0.3
        assert cfg.principal_update_freq == 200
        assert cfg.random_update_freq == 200
        assert cfg.share_subspace is True
        assert cfg.min_dim_threshold == 64

    def test_custom(self):
        cfg = PRACConfig(principal_rank=0.4, random_rank=0.2, principal_update_freq=50)
        assert cfg.principal_rank == 0.4
        assert cfg.random_rank == 0.2
        assert cfg.principal_update_freq == 50


class TestPRACCompressor:
    def test_ranks(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=768)
        assert comp.r1 == int(768 * 0.3)
        assert comp.r2 == int(768 * 0.3)
        assert abs(comp.k - (768 - comp.r1) / comp.r2) < 1e-6

    def test_compress_decompress_shape(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=128)
        x = torch.randn(2, 64, 128)
        XQ1, XQ2 = comp.compress(x)
        assert XQ1.shape == (2, 64, comp.r1)
        assert XQ2.shape == (2, 64, comp.r2)
        x_hat = comp.decompress(XQ1, XQ2)
        assert x_hat.shape == x.shape

    def test_reconstruction_error_bounded(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=128)
        x = torch.randn(2, 64, 128)
        XQ1, XQ2 = comp.compress(x)
        x_hat = comp.decompress(XQ1, XQ2)
        rel_err = torch.norm(x - x_hat) / torch.norm(x)
        assert rel_err < 1.0

    def test_memory_savings(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=100)
        savings = comp.memory_savings()
        expected = 1.0 - (comp.r1 + comp.r2) / 100
        assert abs(savings - expected) < 1e-6

    def test_subspace_orthogonality(self):
        """Q2 should be orthogonal to Q1."""
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=128)
        x = torch.randn(4, 32, 128)
        comp.update_subspaces(x)
        cross = comp.Q1.T @ comp.Q2
        assert cross.abs().max() < 1e-4

    def test_q1_orthonormal(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=128)
        x = torch.randn(4, 32, 128)
        comp.update_subspaces(x)
        eye_approx = comp.Q1.T @ comp.Q1
        assert torch.allclose(eye_approx, torch.eye(comp.r1), atol=1e-4)

    def test_lazy_update_schedule(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3,
                         principal_update_freq=5, random_update_freq=5)
        comp = PRACCompressor(cfg, feature_dim=64)
        x = torch.randn(2, 16, 64)
        comp.update_subspaces(x)
        Q1_first = comp.Q1.clone()
        for _ in range(4):
            comp.update_subspaces(x)
        assert torch.equal(comp.Q1, Q1_first), "Q1 should not change between updates"
        x_new = torch.randn(2, 16, 64)  # different data triggers a different SVD result
        comp.update_subspaces(x_new)  # step 5 -> triggers update
        assert not torch.equal(comp.Q1, Q1_first), "Q1 should update at step 5"


class TestPRACLinear:
    def test_forward_shape(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(128, 256, cfg)
        layer.train()
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 256)

    def test_eval_matches_linear(self):
        """In eval mode, PRACLinear should behave identically to nn.Linear."""
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(128, 256, cfg)
        layer.eval()
        x = torch.randn(2, 16, 128)
        expected = layer.linear(x)
        actual = layer(x)
        assert torch.allclose(actual, expected)

    def test_gradient_flows(self):
        """Gradients should flow through PRACFunction."""
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(128, 64, cfg)
        layer.train()
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.linear.weight.grad is not None

    def test_small_dim_no_compression(self):
        cfg = PRACConfig(min_dim_threshold=100)
        layer = PRACLinear(64, 128, cfg)
        assert layer.compressor is None
        stats = layer.get_memory_stats()
        assert stats["enabled"] is False

    def test_stats(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        layer = PRACLinear(200, 100, cfg)
        stats = layer.get_memory_stats()
        assert stats["enabled"] is True
        assert stats["savings_pct"] > 0


class TestPRACFunction:
    def test_backward_weight_grad_shape(self):
        cfg = PRACConfig(principal_rank=0.3, random_rank=0.3)
        comp = PRACCompressor(cfg, feature_dim=64)
        W = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(32, requires_grad=True)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = PRACFunction.apply(x, W, b, comp)
        loss = out.sum()
        loss.backward()
        assert W.grad.shape == W.shape
        assert b.grad.shape == b.shape
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
