"""
LoRA (Low-Rank Adaptation) adapter configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class LORAConfig(AdapterConfig):
    """LoRA adapter configuration."""

    type: str = "lora"
    r: int = 8  # LoRA rank
    alpha: int = 8  # LoRA alpha (scaling factor)
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])



# Pre-configured LoRA instances for different model sizes
LORA_BASE_CONFIG = LORAConfig(
    r=8,
    alpha=8,
    dropout=0.1,
    target_modules=["query", "value"],
)

LORA_LARGE_CONFIG = LORAConfig(
    r=8,
    alpha=8,
    dropout=0.1,
    target_modules=["query", "value"],
)


# Task-specific hyperparameters for LoRA (from lora.yaml)
LORA_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80, batch_size=32, lr=4e-4, head_lr=1e-3),
    "sst2": TaskConfig(epochs=60, batch_size=16, lr=5e-4, head_lr=1e-3),
    "mrpc": TaskConfig(epochs=30, batch_size=16, lr=4e-4, head_lr=1e-3),
    "qnli": TaskConfig(epochs=25, batch_size=32, lr=4e-4, head_lr=1e-3),
    "rte": TaskConfig(epochs=80, batch_size=32, lr=4e-4, head_lr=1e-3),
    "stsb": TaskConfig(epochs=40, batch_size=16, lr=4e-4, head_lr=1e-3),
}


__all__ = [
    "LORAConfig",
    "LORA_BASE_CONFIG",
    "LORA_LARGE_CONFIG",
    "LORA_TASK_CONFIGS",
]
