"""
LLaMA model adapter with PRAC activation compression support.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict

from src.prac import PRACConfig, apply_prac_to_model, get_prac_stats


def load_llama_with_prac(
    model_name_or_path: str,
    prac_config: Optional[PRACConfig] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    **kwargs,
):
    """
    Load a LLaMA-family model, optionally applying PRAC compression.

    Args:
        model_name_or_path: HF model id or local path.
        prac_config: PRAC configuration (None disables compression).
        load_in_4bit / load_in_8bit: BitsAndBytes quantization flags.
        torch_dtype: Weight dtype.
        device_map: Device mapping strategy.

    Returns:
        The loaded (and optionally PRAC-wrapped) model.
    """
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        **kwargs,
    )

    if prac_config is not None:
        model = apply_prac_to_model(model, prac_config)
        stats = get_prac_stats(model)
        enabled = [s for s in stats.values() if s["enabled"]]
        if enabled:
            avg_savings = sum(s["savings_pct"] for s in enabled) / len(enabled)
            print(
                f"[PRAC] LLaMA: {len(enabled)} layers compressed, "
                f"avg savings {avg_savings:.1f}%"
            )

    return model


def load_llama_tokenizer(model_name_or_path: str, **kwargs):
    """Load a LLaMA tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, **kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer


class LLaMAWithPRAC(nn.Module):
    """Convenience wrapper combining LLaMA with PRAC compression."""

    def __init__(
        self,
        model_name_or_path: str,
        prac_enabled: bool = True,
        principal_rank: float = 0.25,
        random_rank: float = 0.25,
        **load_kwargs,
    ):
        super().__init__()
        prac_config = PRACConfig(
            principal_rank=principal_rank, random_rank=random_rank,
        ) if prac_enabled else None

        self.model = load_llama_with_prac(
            model_name_or_path, prac_config=prac_config, **load_kwargs,
        )
        self.prac_enabled = prac_enabled

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_prac_stats(self) -> Optional[Dict]:
        if not self.prac_enabled:
            return None
        return get_prac_stats(self.model)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
