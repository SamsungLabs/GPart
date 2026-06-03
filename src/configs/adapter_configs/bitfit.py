"""
BitFit adapter configuration (bias-only fine-tuning).
"""

from dataclasses import dataclass
from typing import Any, Dict
from configs.base_config import AdapterConfig, TaskConfig


@dataclass
class BITFITConfig(AdapterConfig):
    """BitFit adapter configuration - bias-only fine-tuning."""

    type: str = "bitfit"
    # No additional parameters - only biases are trained

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PEFT compatibility."""
        return {"type": self.type}


# Pre-configured BitFit instances for different model sizes
BITFIT_BASE_CONFIG = BITFITConfig()
BITFIT_LARGE_CONFIG = BITFITConfig()


# Task-specific hyperparameters for BitFit
BITFIT_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "cola": TaskConfig(epochs=80),
    "sst2": TaskConfig(epochs=60),
    "mrpc": TaskConfig(epochs=30),
    "qnli": TaskConfig(epochs=25),
    "rte": TaskConfig(epochs=80),
    "stsb": TaskConfig(epochs=40),
}


__all__ = [
    "BITFITConfig",
    "BITFIT_BASE_CONFIG",
    "BITFIT_LARGE_CONFIG",
    "BITFIT_TASK_CONFIGS",
]
