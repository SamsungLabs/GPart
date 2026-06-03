"""
Configuration management for GLUE fine-tuning experiments.

This module provides a unified interface for loading and accessing
experiment configurations using a dataclass-based system.

Benefits over YAML + YACS:
- Type safety and IDE autocomplete
- Easier value extraction and manipulation
- Better version control with Python diffs
- No runtime parsing errors from YAML
"""

from typing import Any, Dict, List, Optional

# Import adapter-specific configurations
from configs.adapter_configs import (
    ADAPTER_CONFIG_REGISTRY,
    get_adapter_instance,
    get_adapter_task_configs,
)

# Import base configuration classes
from configs.base_config import (
    AdapterConfig,
    ExperimentConfig,
    TaskConfig,
    TaskMetadata,
    TrainingConfig,
)

# Import task metadata
from configs.task_configs import (
    GLUE_TASK_METADATA,
    get_task_metadata,
)

# Allowed adapters and tasks — derived from their respective registries
# so there is a single source of truth (no need to update in multiple places).
ALLOWED_ADAPTERS = list(ADAPTER_CONFIG_REGISTRY.keys())

ALLOWED_TASKS = list(GLUE_TASK_METADATA.keys())


def get_cfg_defaults():
    """
    Get a default ExperimentConfig object.

    Kept for backward compatibility with existing code.
    """
    return ExperimentConfig()


def load_config(
    adapter_type: str = "lora",
    task_name: Optional[str] = None,
    config_file: Optional[str] = None,
    load_adapter: Optional[str] = None,
    model_size: str = "base",
    opts: Optional[List[str]] = None,
) -> ExperimentConfig:
    """
    Load configuration for an experiment.

    This function creates a complete configuration by combining:
    1. Adapter-specific configuration (from Python dataclasses)
    2. Task-specific hyperparameters (from Python dataclasses)
    3. Task metadata (dataset info, metrics, etc.)
    4. Optional command-line overrides (opts)

    Args:
        adapter_type: Type of adapter (lora, gpart, vera, etc.)
        task_name: Optional task name for task-specific config loading
        config_file: Optional custom config file path (for future extension)
        load_adapter: Optional path to pre-trained adapter
        model_size: Model size variant ("base" or "large")
        opts: Optional command-line overrides in format ["KEY1", "VALUE1", "KEY2", "VALUE2"]

    Returns:
        ExperimentConfig: Complete experiment configuration

    Raises:
        ValueError: If adapter_type or task_name is not supported
    """
    # Validate inputs
    if adapter_type and adapter_type not in ALLOWED_ADAPTERS:
        raise ValueError(
            f"Unsupported adapter type: {adapter_type}. Must be one of {ALLOWED_ADAPTERS}"
        )

    if task_name and task_name not in ALLOWED_TASKS:
        raise ValueError(
            f"Unsupported task name: {task_name}. Must be one of {ALLOWED_TASKS}"
        )

    # Get adapter configuration
    try:
        adapter_config = get_adapter_instance(adapter_type, model_size)
    except ValueError:
        # Fallback to default if adapter not in registry
        adapter_config = AdapterConfig(type=adapter_type)

    # Get task-specific configs for this adapter (with model_size awareness)
    task_configs_dict = {}
    try:
        if adapter_type in ADAPTER_CONFIG_REGISTRY:
            registry = ADAPTER_CONFIG_REGISTRY[adapter_type]
            # Use large_task_configs for large models if available
            if model_size == "large" and "large_task_configs" in registry:
                adapter_task_configs = registry["large_task_configs"]
            else:
                adapter_task_configs = registry["task_configs"]
            task_configs_dict = {adapter_type: adapter_task_configs}
    except (ValueError, KeyError):
        pass  # No task configs for this adapter

    # Create main experiment config
    config = ExperimentConfig(
        adapter=adapter_config,
        training=TrainingConfig(),  # Default training config
        task_metadata=GLUE_TASK_METADATA.copy(),
        task_configs=task_configs_dict,
        load_adapter=load_adapter,
    )

    # Apply command-line overrides if provided
    if opts is not None and len(opts) > 0:
        config = _apply_cli_overrides(config, opts)

    return config


def _apply_cli_overrides(config: ExperimentConfig, opts: List[str]) -> ExperimentConfig:
    """
    Apply command-line overrides to configuration.

    Args:
        config: Base configuration
        opts: List of key-value pairs ["KEY1", "VALUE1", "KEY2", "VALUE2"]

    Returns:
        ExperimentConfig: Configuration with overrides applied
    """
    # Handle potential trailing newlines in the last argument
    opts = list(opts)
    if opts and isinstance(opts[-1], str):
        opts[-1] = opts[-1].strip("\r\n")

    # Parse key-value pairs
    i = 0
    while i < len(opts) - 1:
        key = opts[i]
        value = opts[i + 1]

        # Parse nested keys (e.g., "training.lr" or "adapter.d")
        if "." in key:
            parts = key.split(".", 1)
            section = parts[0]
            subkey = parts[1]

            if section == "training":
                _set_training_attr(config, subkey, value)
            elif section == "adapter":
                _set_adapter_attr(config, subkey, value)
            elif section == "task":
                _set_task_attr(config, subkey, value)

        i += 2

    return config


def _set_training_attr(config: ExperimentConfig, key: str, value: str):
    """Set a training configuration attribute."""
    try:
        # Try to convert to appropriate type
        if hasattr(config.training, key):
            current_value = getattr(config.training, key)
            if isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            elif isinstance(current_value, bool):
                value = value.lower() == "true"
            setattr(config.training, key, value)
    except (ValueError, AttributeError):
        pass  # Ignore invalid overrides


def _set_adapter_attr(config: ExperimentConfig, key: str, value: str):
    """Set an adapter configuration attribute."""
    try:
        # Try to convert to appropriate type
        if hasattr(config.adapter, key):
            current_value = getattr(config.adapter, key)
            if isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            elif isinstance(current_value, bool):
                value = value.lower() == "true"
            setattr(config.adapter, key, value)
    except (ValueError, AttributeError):
        pass  # Ignore invalid overrides


def _set_task_attr(config: ExperimentConfig, key: str, value: str):
    """
    Set a task-specific configuration attribute (e.g., task.lr, task.head_lr).

    This overrides task-specific values in two places to ensure they take
    effect regardless of the fallback chain in get_effective_task_config():
    1. All TaskConfig entries in config.task_configs for the current adapter
    2. The corresponding training config field as a fallback default
    """
    try:
        # Convert value to float for lr/head_lr, int for epochs, etc.
        typed_value: Any = value
        if key in ("lr", "head_lr"):
            typed_value = float(value)
        elif key in ("epochs", "batch_size", "num_labels"):
            typed_value = int(value)
        else:
            # Try float first, then int
            try:
                typed_value = float(value)
                if typed_value.is_integer() and "." not in value:
                    typed_value = int(value)
            except ValueError:
                pass

        # Update all TaskConfig entries for the current adapter
        adapter_type = config.adapter.type
        if adapter_type in config.task_configs:
            for task_name, task_cfg in config.task_configs[adapter_type].items():
                if hasattr(task_cfg, key):
                    setattr(task_cfg, key, typed_value)

        # Also update training config as fallback for tasks not in task_configs
        training_key_map = {
            "lr": "lr",
            "head_lr": "head_lr",
            "batch_size": "batch_size",
        }
        if key in training_key_map and hasattr(config.training, training_key_map[key]):
            setattr(config.training, training_key_map[key], typed_value)

    except (ValueError, AttributeError):
        pass  # Ignore invalid overrides


def get_adapter_config(
    cfg: ExperimentConfig, adapter_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get adapter configuration as a dictionary.

    This function extracts all adapter parameters from the configuration
    object's to_dict() method, which is the single source of truth for
    each adapter type's fields. This avoids duplicating field names in
    a separate ADAPTER_KEYS dict, which was error-prone (e.g., adding
    a field to the config class but forgetting to add it to ADAPTER_KEYS
    would silently drop it).

    Args:
        cfg: Configuration object
        adapter_type: Adapter type (defaults to cfg.adapter.type).
            Kept for API compatibility; the dict comes from the config
            object itself.

    Returns:
        dict: Adapter configuration parameters (excluding 'type' metadata)
    """
    adapter_dict = cfg.adapter.to_dict()
    # Remove 'type' key — it's metadata for routing, not a PEFT parameter
    adapter_dict.pop("type", None)
    return adapter_dict


def get_task_config(
    cfg: ExperimentConfig, task_name: str, adapter_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get task-specific configuration with all fallbacks resolved.

    This method combines:
    1. Task-specific config (from adapter's task_configs)
    2. Task metadata (dataset info, num_labels, etc.)
    3. Training config (as fallback for lr, batch_size, etc.)

    Args:
        cfg: Configuration object
        task_name: Name of the task
        adapter_type: Adapter type for adapter-specific parameters

    Returns:
        dict: Task configuration parameters with all fallbacks resolved
    """
    if adapter_type is None:
        adapter_type = cfg.adapter.type

    # Use the ExperimentConfig's built-in method for merging configs
    return cfg.get_effective_task_config(task_name, adapter_type)


def get_learning_rate(
    cfg: ExperimentConfig, task_name: str, adapter_type: str
) -> float:
    """
    Get the appropriate learning rate for a task and adapter combination.

    Args:
        cfg: Configuration object
        task_name: Name of the task
        adapter_type: Type of adapter

    Returns:
        float: Learning rate
    """
    task_config = get_task_config(cfg, task_name, adapter_type)
    return task_config.get("lr", cfg.training.lr)


__all__ = [
    # Main loading function
    "load_config",
    # Helper functions
    "get_cfg_defaults",
    "get_adapter_config",
    "get_task_config",
    "get_learning_rate",
    # Configuration classes (for advanced usage)
    "ExperimentConfig",
    "AdapterConfig",
    "TrainingConfig",
    "TaskConfig",
    "TaskMetadata",
    # Constants
    "ALLOWED_ADAPTERS",
    "ALLOWED_TASKS",
    "GLUE_TASK_METADATA",
]
