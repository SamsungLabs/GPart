"""
Configuration management for GLUE fine-tuning experiments.

This package provides a dataclass-based configuration system that replaces
the YAML + YACS approach with type-safe, IDE-friendly Python classes.

Usage:
    from configs import load_config, get_adapter_config, get_task_config
    
    # Load configuration
    config = load_config(
        adapter_type="gpart",
        model_size="large",
    )
    
    # Get adapter config as dict (for PEFT)
    adapter_cfg = get_adapter_config(config, "gpart")
    
    # Get task config with all fallbacks resolved
    task_cfg = get_task_config(config, "sst2", "gpart")
    
    # Direct attribute access (IDE-friendly!)
    print(config.adapter.d)  # GPART dimension
    print(config.training.lr)  # Learning rate
"""

# Re-export main configuration functions and classes
from configs.config import (
    ALLOWED_ADAPTERS,
    ALLOWED_TASKS,
    GLUE_TASK_METADATA,
    AdapterConfig,
    ExperimentConfig,
    TaskConfig,
    TaskMetadata,
    TrainingConfig,
    get_adapter_config,
    get_cfg_defaults,
    get_learning_rate,
    get_task_config,
    load_config,
)

# Re-export adapter-specific configurations
from configs.adapter_configs import (
    ADAPTER_CONFIG_REGISTRY,
    BITFIT_BASE_CONFIG,
    BITFIT_LARGE_CONFIG,
    BITFITConfig,
    CONDLORA_BASE_CONFIG,
    CONDLORA_LARGE_CONFIG,
    CONDLORAConfig,
    FULL_BASE_CONFIG,
    FULL_LARGE_CONFIG,
    FULLConfig,
    GPART_BASE_CONFIG,
    GPART_LARGE_CONFIG,
    GPARTConfig,
    HEAD_BASE_CONFIG,
    HEAD_LARGE_CONFIG,
    HEADConfig,
    LORA_BASE_CONFIG,
    LORA_LARGE_CONFIG,
    LORAConfig,
    UNILORA_BASE_CONFIG,
    UNILORA_LARGE_CONFIG,
    UNILORAConfig,
    VERA_BASE_CONFIG,
    VERA_LARGE_CONFIG,
    VERAConfig,
    get_adapter_instance,
    get_adapter_task_configs,
)

# Re-export task metadata
from configs.task_configs import (
    REGRESSION_TASKS,
    TASK_PRIMARY_METRICS,
    get_all_task_metadata,
    get_task_metadata,
    GLUE_TASK_METADATA,
)

__all__ = [
    # Main API
    "load_config",
    "get_cfg_defaults",
    "get_adapter_config",
    "get_task_config",
    "get_learning_rate",
    "get_adapter_instance",
    "get_adapter_task_configs",
    "get_task_metadata",
    "get_all_task_metadata",
    # Configuration classes
    "ExperimentConfig",
    "AdapterConfig",
    "TrainingConfig",
    "TaskConfig",
    "TaskMetadata",
    # Adapter configs
    "ADAPTER_CONFIG_REGISTRY",
    "LORAConfig",
    "LORA_BASE_CONFIG",
    "LORA_LARGE_CONFIG",
    "GPARTConfig",
    "GPART_BASE_CONFIG",
    "GPART_LARGE_CONFIG",
    "VERAConfig",
    "VERA_BASE_CONFIG",
    "VERA_LARGE_CONFIG",
    "UNILORAConfig",
    "UNILORA_BASE_CONFIG",
    "UNILORA_LARGE_CONFIG",
    "CONDLORAConfig",
    "CONDLORA_BASE_CONFIG",
    "CONDLORA_LARGE_CONFIG",
    "BITFITConfig",
    "BITFIT_BASE_CONFIG",
    "BITFIT_LARGE_CONFIG",
    "HEADConfig",
    "HEAD_BASE_CONFIG",
    "HEAD_LARGE_CONFIG",
    "FULLConfig",
    "FULL_BASE_CONFIG",
    "FULL_LARGE_CONFIG",
    # Constants
    "ALLOWED_ADAPTERS",
    "ALLOWED_TASKS",
    "GLUE_TASK_METADATA",
    "TASK_PRIMARY_METRICS",
    "REGRESSION_TASKS",
]
