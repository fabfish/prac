"""
PRAC: Principal-Random Subspace for LLM Activation Compression
Core implementation module.

Paper: PRAC: Principal-Random Subspace for LLM Activation Compression
       and Memory-Efficient Training (arXiv:2602.23111)
Authors: Yanyi Li, Yimu Zhang, Cong Fang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PRACConfig:
    """PRAC configuration."""
    principal_rank: float = 0.3
    random_rank: float = 0.3
    scaling_factor: Optional[float] = None
    principal_update_freq: int = 200
    random_update_freq: int = 200
    share_subspace: bool = True
    layer_configs: Optional[Dict[str, Tuple[float, float]]] = None
    min_dim_threshold: int = 64


class PRACCompressor:
    """
    PRAC activation compressor.

    Core algorithm:
    1. SVD on activation X to get principal subspace Q1 (top-r1 right singular vectors)
    2. Sample Q2 uniformly from the orthogonal complement of Q1
    3. Store compressed projections: XQ1 (size b*r1) and XQ2 (size b*r2)
    4. Reconstruct: X_hat = (XQ1)Q1^T + k*(XQ2)Q2^T,  k = (n - r1) / r2
    """

    def __init__(self, config: PRACConfig, feature_dim: int):
        self.config = config
        self.feature_dim = feature_dim

        self.r1 = max(1, int(feature_dim * config.principal_rank))
        self.r2 = max(1, int(feature_dim * config.random_rank))

        if config.scaling_factor is None:
            self.k = (feature_dim - self.r1) / self.r2
        else:
            self.k = config.scaling_factor

        self.Q1: Optional[torch.Tensor] = None  # [n, r1]
        self.Q2: Optional[torch.Tensor] = None  # [n, r2]
        self.step_count = 0

    def update_subspaces(self, activations: torch.Tensor):
        """Update subspace projection matrices on schedule."""
        should_update_principal = (
            self.step_count % self.config.principal_update_freq == 0
            or self.Q1 is None
        )
        should_update_random = (
            self.step_count % self.config.random_update_freq == 0
            or self.Q2 is None
        )

        if should_update_principal:
            self._update_principal_subspace(activations)

        if should_update_random:
            self._update_random_subspace()

        self.step_count += 1

    def _update_principal_subspace(self, activations: torch.Tensor):
        """Update principal subspace via truncated SVD."""
        if activations.dim() == 3:
            X = activations.reshape(-1, activations.shape[-1])
        else:
            X = activations

        orig_dtype = activations.dtype
        try:
            _, _, Vh = torch.linalg.svd(X.float(), full_matrices=False)
            self.Q1 = Vh[:self.r1, :].T.contiguous().detach().to(orig_dtype)
        except Exception:
            Q = torch.randn(
                self.feature_dim, self.r1,
                device=activations.device, dtype=torch.float32,
            )
            Q1_f, _ = torch.linalg.qr(Q)
            self.Q1 = Q1_f.to(orig_dtype)

    def _update_random_subspace(self):
        """Sample Q2 uniformly from the orthogonal complement of Q1."""
        if self.Q1 is None:
            raise RuntimeError("Principal subspace must be initialized before random subspace.")

        device = self.Q1.device
        orig_dtype = self.Q1.dtype
        Q1_f = self.Q1.float()
        Z = torch.randn(self.feature_dim, self.r2, device=device, dtype=torch.float32)
        Z = Z - Q1_f @ (Q1_f.T @ Z)
        Q2_f, _ = torch.linalg.qr(Z)
        self.Q2 = Q2_f.to(orig_dtype)

    def compress(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress activations by projecting onto principal + random subspaces.

        Returns:
            XQ1: [B, S, r1]  principal projection
            XQ2: [B, S, r2]  random projection
        """
        if self.Q1 is None or self.Q2 is None:
            self.update_subspaces(activations)

        Q1 = self.Q1.to(device=activations.device, dtype=activations.dtype)
        Q2 = self.Q2.to(device=activations.device, dtype=activations.dtype)

        XQ1 = activations @ Q1
        XQ2 = activations @ Q2
        return XQ1, XQ2

    def decompress(self, XQ1: torch.Tensor, XQ2: torch.Tensor) -> torch.Tensor:
        """Reconstruct activation: X_hat = XQ1 @ Q1^T + k * XQ2 @ Q2^T"""
        Q1 = self.Q1.to(device=XQ1.device, dtype=XQ1.dtype)
        Q2 = self.Q2.to(device=XQ1.device, dtype=XQ1.dtype)
        return XQ1 @ Q1.T + self.k * (XQ2 @ Q2.T)

    def memory_savings(self) -> float:
        return 1.0 - (self.r1 + self.r2) / self.feature_dim


class PRACFunction(torch.autograd.Function):
    """
    Custom autograd function that saves only compressed activations for backward.

    Forward:  y = x @ W^T + b           (exact, using original x)
    Backward: grad_W = grad_y^T @ x_hat  (approximate, using reconstructed x_hat)
              grad_x = grad_y @ W        (exact)

    This is the key to PRAC's memory savings: instead of storing the full
    activation tensor x (size B*S*n), we store XQ1 (B*S*r1) + XQ2 (B*S*r2).
    """

    @staticmethod
    def forward(ctx, x, weight, bias, compressor):
        XQ1, XQ2 = compressor.compress(x)
        ctx.save_for_backward(XQ1, XQ2, weight)
        ctx.has_bias = bias is not None
        ctx.compressor = compressor
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        XQ1, XQ2, weight = ctx.saved_tensors
        compressor = ctx.compressor

        x_hat = compressor.decompress(XQ1, XQ2)

        compute_dtype = grad_output.dtype
        grad_input = grad_output @ weight.to(compute_dtype)

        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        xh_2d = x_hat.reshape(-1, x_hat.shape[-1]).to(compute_dtype)
        grad_weight = (go_2d.T @ xh_2d).to(weight.dtype)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = go_2d.sum(0).to(weight.dtype)

        return grad_input, grad_weight, grad_bias, None


class PRACLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with PRAC activation compression.

    During training, uses PRACFunction to save only compressed activations
    for the backward pass, reducing peak memory.  During eval, behaves
    identically to nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int,
                 config: PRACConfig, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.compressor: Optional[PRACCompressor] = None
        if in_features >= config.min_dim_threshold:
            self.compressor = PRACCompressor(config, in_features)

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        self.linear.bias = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compressor is not None and self.training:
            self.compressor.update_subspaces(x.detach())
            return PRACFunction.apply(
                x, self.linear.weight, self.linear.bias, self.compressor
            )
        return self.linear(x)

    def get_memory_stats(self) -> Dict:
        if self.compressor is None:
            return {
                "original_dim": self.in_features,
                "compressed_dim": self.in_features,
                "savings_pct": 0.0,
                "enabled": False,
            }
        return {
            "original_dim": self.in_features,
            "compressed_dim": self.compressor.r1 + self.compressor.r2,
            "savings_pct": self.compressor.memory_savings() * 100,
            "enabled": True,
            "principal_rank": self.compressor.r1,
            "random_rank": self.compressor.r2,
        }


def apply_prac_to_model(model: nn.Module, config: PRACConfig) -> nn.Module:
    """
    Replace qualifying nn.Linear layers in *model* with PRACLinear.

    Shares weight tensors (no copy) so model size does not increase.
    Call this BEFORE applying LoRA/PEFT so that PEFT can still discover
    the inner ``self.linear`` (an nn.Linear) via target_modules.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features >= config.min_dim_threshold:
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        prac_linear = PRACLinear(
            module.in_features, module.out_features,
            config, bias=module.bias is not None,
        )
        # Share weight tensors -- no copy
        prac_linear.linear.weight = module.weight
        if module.bias is not None:
            prac_linear.linear.bias = module.bias

        setattr(parent, child_name, prac_linear)

    return model


def get_prac_stats(model: nn.Module) -> Dict[str, Dict]:
    """Collect PRAC memory stats from all PRACLinear layers."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, PRACLinear):
            stats[name] = module.get_memory_stats()
    return stats


if __name__ == "__main__":
    config = PRACConfig(principal_rank=0.3, random_rank=0.3)
    compressor = PRACCompressor(config, feature_dim=768)

    x = torch.randn(2, 128, 768)
    XQ1, XQ2 = compressor.compress(x)
    print(f"Original: {x.shape}, memory: {x.numel()}")
    print(f"Compressed: XQ1 {XQ1.shape}, XQ2 {XQ2.shape}, total: {XQ1.numel() + XQ2.numel()}")
    print(f"Savings: {compressor.memory_savings() * 100:.1f}%")

    x_recon = compressor.decompress(XQ1, XQ2)
    error = torch.norm(x - x_recon) / torch.norm(x)
    print(f"Reconstruction error: {error.item():.4f}")
