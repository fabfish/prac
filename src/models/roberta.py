"""
RoBERTa model adapter with PRAC activation compression support.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Optional, Dict

from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats


def load_roberta_with_prac(
    model_name: str,
    num_labels: int = 2,
    prac_config: Optional[PRACConfig] = None,
    **kwargs,
):
    """Load a RoBERTa model, optionally applying PRAC compression."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, **kwargs,
    )

    if prac_config is not None:
        model = apply_prac_to_model(model, prac_config)
        stats = get_prac_stats(model)
        enabled = [s for s in stats.values() if s["enabled"]]
        if enabled:
            avg_savings = sum(s["savings_pct"] for s in enabled) / len(enabled)
            print(f"[PRAC] RoBERTa: {len(enabled)} layers compressed, avg savings {avg_savings:.1f}%")

    return model


class RoBERTaWithPRAC(nn.Module):
    """Convenience wrapper: RoBERTa + PRAC."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        prac_enabled: bool = True,
        principal_rank: float = 0.3,
        random_rank: float = 0.3,
    ):
        super().__init__()
        prac_config = PRACConfig(
            principal_rank=principal_rank, random_rank=random_rank,
        ) if prac_enabled else None

        self.model = load_roberta_with_prac(
            model_name, num_labels=num_labels, prac_config=prac_config,
        )
        self.prac_enabled = prac_enabled

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_prac_stats(self) -> Optional[Dict]:
        if not self.prac_enabled:
            return None
        return get_prac_stats(self.model)

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
