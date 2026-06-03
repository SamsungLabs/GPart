"""
Base configuration dataclasses for GLUE fine-tuning experiments.

This module provides type-safe, IDE-friendly configuration classes
to replace the YAML + YACS system.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    batch_size: int = 32
    max_seq_length: int = 512
    weight_decay: float = 0.1
    warmup_ratio: float = 0.06
    model_selection: str = "best"  # "best" = best val score; "last" = last epoch (no val split)
    default_num_seeds: int = 1
    lr: float = 1e-3
    head_lr: float = 1e-3


@dataclass
class TaskMetadata:
    """Task-specific metadata for dataset loading and evaluation."""

    dataset: Tuple[str, str] = field(default_factory=lambda: ("nyu-mll/glue", "cola"))
    num_labels: int = 2
    text_keys: List[Optional[str]] = field(default_factory=lambda: ["sentence", None])
    metric_fn: str = "accuracy"
    split: str = "validation"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "dataset": self.dataset,
            "num_labels": self.num_labels,
            "text_keys": self.text_keys,
            "metric_fn": self.metric_fn,
            "split": self.split,
        }


@dataclass
class TaskConfig:
    """Task-specific hyperparameters."""

    epochs: int = 10
    batch_size: Optional[int] = None  # None means use training.batch_size
    lr: Optional[float] = None  # None means use training.lr
    head_lr: Optional[float] = None  # None means use training.head_lr
    dataset: Optional[Tuple[str, str]] = None
    num_labels: Optional[int] = None
    text_keys: Optional[List[Optional[str]]] = None
    metric_fn: Optional[str] = None
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "head_lr": self.head_lr,
            "dataset": self.dataset,
            "num_labels": self.num_labels,
            "text_keys": self.text_keys,
            "metric_fn": self.metric_fn,
            "split": self.split,
        }


@dataclass
class AdapterConfig:
    """Base adapter configuration - to be extended by specific adapters."""

    type: str = "lora"
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with PEFT.

        Uses dataclasses.asdict() so that subclass fields are automatically
        included — no need to manually list field names or call super().
        Subclasses that want to exclude inherited fields (e.g. HEADConfig)
        can override this method.
        """
        return asdict(self)


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration container.

    This class provides a unified interface for accessing all configuration
    parameters needed for GLUE fine-tuning experiments.
    """

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task_metadata: Dict[str, TaskMetadata] = field(default_factory=dict)
    task_configs: Dict[str, Dict[str, TaskConfig]] = field(default_factory=dict)
    load_adapter: Optional[str] = None

    def get_task_metadata(self, task_name: str) -> TaskMetadata:
        """Get task metadata, falling back to defaults."""
        if task_name in self.task_metadata:
            return self.task_metadata[task_name]
        # Return default metadata
        return TaskMetadata()

    def get_task_config(self, task_name: str, adapter_type: str) -> TaskConfig:
        """Get task-specific config for a given adapter type."""
        if (
            adapter_type in self.task_configs
            and task_name in self.task_configs[adapter_type]
        ):
            return self.task_configs[adapter_type][task_name]
        # Return default task config
        return TaskConfig()

    def get_effective_task_config(
        self, task_name: str, adapter_type: str
    ) -> Dict[str, Any]:
        """
        Get merged task configuration with all fallbacks resolved.

        This method combines:
        1. Task-specific config (from adapter's task_configs)
        2. Task metadata (dataset info, num_labels, etc.)
        3. Training config (as fallback for lr, batch_size, etc.)

        Returns a dictionary compatible with the existing code.
        """
        task_config = self.get_task_config(task_name, adapter_type)
        metadata = self.get_task_metadata(task_name)
        training = self.training

        # Build merged config with proper fallbacks
        result = {
            "dataset": task_config.dataset
            or metadata.dataset
            or ("nyu-mll/glue", task_name),
            "num_labels": task_config.num_labels or metadata.num_labels or 2,
            "text_keys": task_config.text_keys
            or metadata.text_keys
            or ["sentence", None],
            "metric_fn": task_config.metric_fn or metadata.metric_fn or "accuracy",
            "split": task_config.split or metadata.split or "validation",
            "epochs": task_config.epochs,
            "batch_size": task_config.batch_size or training.batch_size,
            "lr": task_config.lr or training.lr,
            "head_lr": task_config.head_lr or training.head_lr,
        }

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            "adapter": self.adapter.to_dict(),
            "training": asdict(self.training),
            "load_adapter": self.load_adapter,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backward compatibility."""
        if key == "adapter":
            return self.adapter.to_dict()
        elif key == "training":
            return asdict(self.training)
        elif key == "task":
            return {
                "task_metadata": {
                    k: v.to_dict() for k, v in self.task_metadata.items()
                },
            }
        elif key == "load_adapter":
            return self.load_adapter
        raise KeyError(f"Unknown config key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-like get() for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

    def has_key(self, key: str) -> bool:
        """Check if config has a key."""
        return key in ["adapter", "training", "task", "load_adapter"]
