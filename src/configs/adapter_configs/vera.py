"""
VeRA (Vector-based Randomized Adaptation) adapter configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class VERAConfig(AdapterConfig):
    """VeRA adapter configuration."""

    type: str = "vera"
    r: int = 8  # VeRA rank
    dropout: float = 0.1
    d_initial: float = 0.1  # Initial scaling factor
    projection_prng_key: int = 0  # PRNG key for projection
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])



# Pre-configured VeRA instances for different model sizes
VERA_BASE_CONFIG = VERAConfig(
    r=8,
    dropout=0.1,
    d_initial=0.1,
    projection_prng_key=0,
    target_modules=["query", "value"],
)

VERA_LARGE_CONFIG = VERAConfig(
    r=8,
    dropout=0.1,
    d_initial=0.1,
    projection_prng_key=0,
    target_modules=["query", "value"],
)


# Task-specific hyperparameters for VeRA
VERA_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80),
    "sst2": TaskConfig(epochs=60),
    "mrpc": TaskConfig(epochs=30),
    "qnli": TaskConfig(epochs=25),
    "rte": TaskConfig(epochs=80),
    "stsb": TaskConfig(epochs=40),
}


__all__ = [
    "VERAConfig",
    "VERA_BASE_CONFIG",
    "VERA_LARGE_CONFIG",
    "VERA_TASK_CONFIGS",
]
