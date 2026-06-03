"""
UniLoRA (Universal Low-Rank Adaptation) adapter configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class UNILORAConfig(AdapterConfig):
    """UniLoRA adapter configuration."""

    type: str = "unilora"
    r: int = 4  # LoRA rank
    theta_d_length: int = 23040  # Theta dimension length
    dropout: float = 0.1
    init_theta_d_bound: float = 0.02  # Initialization bound for theta
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])


# Pre-configured UniLoRA instances for different model sizes
UNILORA_BASE_CONFIG = UNILORAConfig(
    r=4,
    theta_d_length=23040,
    dropout=0.1,
    init_theta_d_bound=0.02,
    target_modules=["query", "value"],
)

UNILORA_LARGE_CONFIG = UNILORAConfig(
    r=4,
    theta_d_length=23040,
    dropout=0.1,
    init_theta_d_bound=0.02,
    target_modules=["query", "value"],
)


# Task-specific hyperparameters for UniLoRA
UNILORA_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80, lr=5e-3, head_lr=5e-3),
    "sst2": TaskConfig(epochs=60, lr=5e-3, head_lr=1e-4),
    "mrpc": TaskConfig(epochs=30, lr=5e-3, head_lr=2e-2),
    "qnli": TaskConfig(epochs=25, lr=5e-3, head_lr=2e-4),
    "rte": TaskConfig(epochs=80, lr=5e-3, head_lr=5e-4),
    "stsb": TaskConfig(epochs=40, lr=5e-3, head_lr=2e-4),
}


UNILORA_LARGE_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=40, lr=5e-3, head_lr=2e-2),
    "sst2": TaskConfig(epochs=20, lr=5e-3, head_lr=2e-4),
    "mrpc": TaskConfig(epochs=40, lr=5e-3, head_lr=2e-3),
    "qnli": TaskConfig(epochs=20, lr=5e-3, head_lr=5e-3),
    "rte": TaskConfig(epochs=40, lr=5e-3, head_lr=5e-3),
    "stsb": TaskConfig(epochs=40, lr=5e-3, head_lr=1e-4),
}


__all__ = [
    "UNILORAConfig",
    "UNILORA_BASE_CONFIG",
    "UNILORA_LARGE_CONFIG",
    "UNILORA_TASK_CONFIGS",
    "UNILORA_LARGE_TASK_CONFIGS",
]
