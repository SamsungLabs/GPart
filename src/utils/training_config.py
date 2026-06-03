"""
Training configuration dataclasses for GLUE fine-tuning experiments.

This module provides structured configuration classes that replace
the scattered parameter passing in the original script.
"""

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunTaskConfig:
    """
    Configuration for running a single task fine-tuning job.

    Stores the ExperimentConfig directly instead of duplicating training
    hyperparameters. Resolved values (with task-specific overrides applied)
    are available via cached_property — computed once on first access.

    This ensures a single source of truth: adding a new field to
    TrainingConfig automatically makes it available here without any
    manual updates.
    """

    # Core parameters
    task_name: str
    base_model_name: str
    output_root: str
    seed: int
    adapter_type: str

    # The full config object — single source of truth
    config: Any = None  # ExperimentConfig (Any to avoid circular import)

    # Adapter-specific parameters (not in ExperimentConfig)
    main_seed: Optional[int] = None  # For GPart reproducibility
    load_adapter: Optional[str] = None  # Path to pre-trained adapter

    # Flags
    compute_params_only: bool = False

    # Optional MLflow run context
    task_mlflow_run: Any = None

    # ── Resolved config properties ──────────────────────────────
    # These merge task-specific overrides with TrainingConfig defaults.
    # Computed once on first access, then cached.

    @cached_property
    def adapter_config(self) -> Dict[str, Any]:
        """Adapter config as a dict (type field excluded)."""
        from configs.config import get_adapter_config

        return get_adapter_config(self.config, self.adapter_type)

    @cached_property
    def task_config(self) -> Dict[str, Any]:
        """Resolved task config with all fallbacks applied."""
        from configs.config import get_task_config

        return get_task_config(self.config, self.task_name, self.adapter_type)

    @cached_property
    def head_lr(self) -> float:
        """Resolved head LR: task_config > training config."""
        return float(
            self.task_config.get("head_lr", self.config.training.head_lr)
        )

    @cached_property
    def base_lr(self) -> float:
        """Resolved base LR: task_config > training config."""
        return float(self.task_config.get("lr", self.config.training.lr))

    @cached_property
    def batch_size(self) -> int:
        """Resolved batch size: task_config > training config."""
        return self.task_config.get(
            "batch_size", self.config.training.batch_size
        )

    @cached_property
    def max_seq_length(self) -> int:
        """Max sequence length from training config."""
        return self.config.training.max_seq_length

    @cached_property
    def epochs(self) -> int:
        """Number of epochs from task config."""
        return self.task_config.get("epochs", 10)

    @cached_property
    def weight_decay(self) -> float:
        """Weight decay from training config."""
        return self.config.training.weight_decay

    @cached_property
    def warmup_ratio(self) -> float:
        """Warmup ratio from training config."""
        return self.config.training.warmup_ratio

    @cached_property
    def model_selection(self) -> str:
        """Model selection strategy: 'best' or 'last'."""
        return self.config.training.model_selection

    @cached_property
    def target_modules(self) -> List[str]:
        """Target modules from adapter config."""
        return self.adapter_config.get(
            "target_modules", self.config.adapter.target_modules
        )

    # ── Factory method ──────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        task_name: str,
        base_model_name: str,
        output_root: str,
        seed: int,
        adapter_type: str,
        config,
        compute_params_only: bool = False,
        task_mlflow_run: Any = None,
        **kwargs,
    ) -> "RunTaskConfig":
        """
        Create RunTaskConfig from the config system objects.

        This factory method stores the ExperimentConfig directly —
        resolved values are computed lazily via cached_property.
        """
        # Pass main_seed for GPart reproducibility
        main_seed = kwargs.get("main_seed", seed if adapter_type == "gpart" else None)
        load_adapter = kwargs.get("load_adapter", None)

        return cls(
            task_name=task_name,
            base_model_name=base_model_name,
            output_root=output_root,
            seed=seed,
            adapter_type=adapter_type,
            config=config,
            main_seed=main_seed,
            load_adapter=load_adapter,
            compute_params_only=compute_params_only,
            task_mlflow_run=task_mlflow_run,
        )

    # ── Convenience methods ──────────────────────────────────────

    def get_val_batch_size(self) -> int:
        """Get validation batch size using VAL_BATCH_SIZE_MULTIPLIER."""
        from .data_loader_utils import VAL_BATCH_SIZE_MULTIPLIER

        return self.batch_size * VAL_BATCH_SIZE_MULTIPLIER

    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        return int((dataset_size / self.batch_size) * self.epochs)

    def get_warmup_steps(self, total_steps: int) -> int:
        """Calculate warmup steps."""
        return int(total_steps * self.warmup_ratio)

    def is_regression_task(self) -> bool:
        """Check if this is a regression task."""
        return self.task_config.get("num_labels", 2) == 1

    def get_dataset_info(self) -> Tuple[str, str]:
        """Get dataset path and config."""
        dataset = self.task_config.get("dataset", ("nyu-mll/glue", self.task_name))
        return dataset[0], dataset[1]

    def get_text_keys(self) -> List[Optional[str]]:
        """Get text keys for tokenization."""
        return self.task_config.get("text_keys", ["sentence", None])

    def get_eval_split(self) -> str:
        """Get evaluation split name."""
        return self.task_config.get("split", "validation")

    def get_num_labels(self) -> int:
        """Get number of labels."""
        return self.task_config.get("num_labels", 2)


@dataclass
class ExperimentArgs:
    """
    Top-level CLI arguments for the entire GLUE experiment.

    This dataclass holds all parameters parsed from command-line arguments
    in the main() function. Distinct from configs.base_config.ExperimentConfig
    which holds the config-system experiment configuration.
    """

    adapter_type: str
    model_size: str
    base_model: str
    tasks: List[str]
    seed: int
    output_dir: str
    compute_params_only: bool

    # Optional parameters
    load_adapter: Optional[str] = None
    config_file: Optional[str] = None
    opts: Optional[List[str]] = None

    @classmethod
    def from_args(cls, args) -> "ExperimentArgs":
        """Create ExperimentArgs from parsed argparse arguments."""
        # Determine base model
        base_model = args.base_model
        if base_model is None:
            base_model = f"roberta-{args.model_size}"

        # Update output directory to include model size if using default
        subdir = f"roberta_{args.model_size}_glue_{args.adapter_type}"
        output_dir = os.path.join(args.output_dir, subdir)

        return cls(
            adapter_type=args.adapter_type,
            model_size=args.model_size,
            base_model=base_model,
            tasks=args.tasks,
            seed=args.seed,
            output_dir=output_dir,
            compute_params_only=args.compute_params_only,
            load_adapter=args.load_adapter,
            config_file=args.config_file,
            opts=args.opts,
        )
