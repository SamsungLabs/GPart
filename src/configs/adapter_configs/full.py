"""
Full fine-tuning configuration (all parameters trainable).
"""

from dataclasses import dataclass
from typing import Any, Dict
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class FULLConfig(AdapterConfig):
    """Full fine-tuning configuration - all parameters are trainable."""

    type: str = "full"
    # No additional parameters - all model parameters are trained

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PEFT compatibility."""
        return {"type": self.type}


# Pre-configured Full fine-tuning instances for different model sizes
FULL_BASE_CONFIG = FULLConfig()
FULL_LARGE_CONFIG = FULLConfig()


# Task-specific hyperparameters for Full fine-tuning
FULL_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80),
    "sst2": TaskConfig(epochs=60),
    "mrpc": TaskConfig(epochs=30),
    "qnli": TaskConfig(epochs=25),
    "rte": TaskConfig(epochs=80),
    "stsb": TaskConfig(epochs=40),
}


__all__ = [
    "FULLConfig",
    "FULL_BASE_CONFIG",
    "FULL_LARGE_CONFIG",
    "FULL_TASK_CONFIGS",
]
