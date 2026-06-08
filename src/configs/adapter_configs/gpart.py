"""
GPART adapter configuration.

GPART (Graph-based Parameter-efficient Adaptive Residual Tuning) uses
random grouping of parameters with shared low-rank adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class GPARTConfig(AdapterConfig):
    """GPART adapter configuration."""

    type: str = "gpart"
    d: int = 23040  # Dimension of the shared low-rank matrix
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    init_bound: float = 0.0  # Initialization bound for theta parameters
    isometric: bool = True  # Use isometric initialization
    grouping_strategy: Literal["random", "signed_magnitude"] = "random"
    bias: str = (
        "none"  # "none" = exclude biases from partition; "all"/"gpart_only" = include
    )


# Pre-configured GPART instances for different model sizes
GPART_BASE_CONFIG = GPARTConfig(
    d=23040,
    dropout=0.1,
    target_modules=["query", "value"],
    init_bound=0.0,
    isometric=True,
    grouping_strategy="random",
    bias="none",
)

GPART_LARGE_CONFIG = GPARTConfig(
    d=23040,
    dropout=0.1,
    target_modules=["query", "value"],
    init_bound=0.0,
    isometric=True,
    grouping_strategy="random",
    bias="none",
)


# Task-specific hyperparameters for GPART
# These are extracted from gpart.yaml and gpart-large.yaml
GPART_TASK_CONFIGS: Dict[str, TaskConfig] = {
    # Base model task configs (from gpart.yaml)
    "cola": TaskConfig(epochs=80, lr=5e-3, head_lr=2e-3),
    "sst2": TaskConfig(epochs=60, lr=5e-3, head_lr=5e-4),
    "mrpc": TaskConfig(epochs=30, lr=5e-3, head_lr=1e-3),
    "qnli": TaskConfig(epochs=25, lr=5e-3, head_lr=1e-3),
    "rte": TaskConfig(epochs=160, lr=5e-3, head_lr=1e-2),
    "stsb": TaskConfig(epochs=80, lr=5e-3, head_lr=2e-4),
}

# Large model task configs override (from gpart-large.yaml)
GPART_LARGE_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=60, lr=5e-3, head_lr=2e-2),
    "sst2": TaskConfig(epochs=30, lr=5e-3, head_lr=2e-4),
    "mrpc": TaskConfig(epochs=40, lr=5e-3, head_lr=2e-3),
    "qnli": TaskConfig(epochs=25, lr=5e-3, head_lr=5e-3),
    "rte": TaskConfig(epochs=120, lr=5e-3, head_lr=5e-3),
    "stsb": TaskConfig(epochs=80, lr=5e-3, head_lr=1e-4),
}

__all__ = [
    "GPARTConfig",
    "GPART_BASE_CONFIG",
    "GPART_LARGE_CONFIG",
    "GPART_TASK_CONFIGS",
    "GPART_LARGE_TASK_CONFIGS",
]
