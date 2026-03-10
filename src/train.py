"""
PRAC-aware Trainer wrapper built on HuggingFace Trainer.
"""

import torch
from transformers import Trainer
from typing import Optional

from src.prac import PRACConfig, get_prac_stats


class PRACTrainer(Trainer):
    """HuggingFace Trainer with periodic PRAC memory-savings logging."""

    def __init__(self, prac_config: Optional[PRACConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prac_config = prac_config
        self._prac_log_interval = 50

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if (
            self.prac_config is not None
            and self.state.global_step % self._prac_log_interval == 0
        ):
            stats = get_prac_stats(model)
            if stats:
                enabled = [s for s in stats.values() if s["enabled"]]
                if enabled:
                    avg = sum(s["savings_pct"] for s in enabled) / len(enabled)
                    self.log({"prac/avg_savings_pct": avg})

        return loss
