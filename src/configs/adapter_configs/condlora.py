"""
CondLoRA (Conditional Low-Rank Adaptation) adapter configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class CONDLORAConfig(AdapterConfig):
    """CondLoRA adapter configuration."""

    type: str = "condlora"
    r: int = 8  # LoRA rank
    alpha: int = 8  # LoRA alpha
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])



# Pre-configured CondLoRA instances for different model sizes
CONDLORA_BASE_CONFIG = CONDLORAConfig(
    r=8,
    alpha=8,
    dropout=0.1,
    target_modules=["query", "value"],
)

CONDLORA_LARGE_CONFIG = CONDLORAConfig(
    r=8,
    alpha=8,
    dropout=0.1,
    target_modules=["query", "value"],
)


# Task-specific hyperparameters for CondLoRA
CONDLORA_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80),
    "sst2": TaskConfig(epochs=60),
    "mrpc": TaskConfig(epochs=30),
    "qnli": TaskConfig(epochs=25),
    "rte": TaskConfig(epochs=80),
    "stsb": TaskConfig(epochs=40),
}


__all__ = [
    "CONDLORAConfig",
    "CONDLORA_BASE_CONFIG",
    "CONDLORA_LARGE_CONFIG",
    "CONDLORA_TASK_CONFIGS",
]
