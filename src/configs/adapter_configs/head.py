"""
Head-only adapter configuration (classifier-only fine-tuning).
"""

from dataclasses import dataclass
from typing import Any, Dict
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class HEADConfig(AdapterConfig):
    """Head-only adapter configuration - only classifier layer is trained."""

    type: str = "head"
    # No additional parameters - only classifier head is trained

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PEFT compatibility."""
        return {"type": self.type}


# Pre-configured Head-only instances for different model sizes
HEAD_BASE_CONFIG = HEADConfig()
HEAD_LARGE_CONFIG = HEADConfig()


# Task-specific hyperparameters for Head-only
HEAD_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80),
    "sst2": TaskConfig(epochs=60),
    "mrpc": TaskConfig(epochs=30),
    "qnli": TaskConfig(epochs=25),
    "rte": TaskConfig(epochs=80),
    "stsb": TaskConfig(epochs=40),
}


__all__ = [
    "HEADConfig",
    "HEAD_BASE_CONFIG",
    "HEAD_LARGE_CONFIG",
    "HEAD_TASK_CONFIGS",
]
