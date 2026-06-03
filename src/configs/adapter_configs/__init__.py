"""
Adapter-specific configurations for GLUE fine-tuning experiments.

Each adapter type has its own configuration class and predefined instances
for different model sizes (base, large).
"""

from .lora import LORAConfig, LORA_BASE_CONFIG, LORA_LARGE_CONFIG, LORA_TASK_CONFIGS
from .gpart import (
    GPARTConfig,
    GPART_BASE_CONFIG,
    GPART_LARGE_CONFIG,
    GPART_TASK_CONFIGS,
    GPART_LARGE_TASK_CONFIGS,
)
from .vera import VERAConfig, VERA_BASE_CONFIG, VERA_LARGE_CONFIG, VERA_TASK_CONFIGS
from .unilora import UNILORAConfig, UNILORA_BASE_CONFIG, UNILORA_LARGE_CONFIG, UNILORA_TASK_CONFIGS, UNILORA_LARGE_TASK_CONFIGS
from .condlora import CONDLORAConfig, CONDLORA_BASE_CONFIG, CONDLORA_LARGE_CONFIG, CONDLORA_TASK_CONFIGS
from .bitfit import BITFITConfig, BITFIT_BASE_CONFIG, BITFIT_LARGE_CONFIG, BITFIT_TASK_CONFIGS
from .head import HEADConfig, HEAD_BASE_CONFIG, HEAD_LARGE_CONFIG, HEAD_TASK_CONFIGS
from .full import FULLConfig, FULL_BASE_CONFIG, FULL_LARGE_CONFIG, FULL_TASK_CONFIGS

# Mapping from adapter type to config classes and instances
ADAPTER_CONFIG_REGISTRY = {
    "lora": {
        "config_class": LORAConfig,
        "base": LORA_BASE_CONFIG,
        "large": LORA_LARGE_CONFIG,
        "task_configs": LORA_TASK_CONFIGS,
    },
    "gpart": {
        "config_class": GPARTConfig,
        "base": GPART_BASE_CONFIG,
        "large": GPART_LARGE_CONFIG,
        "task_configs": GPART_TASK_CONFIGS,
        "large_task_configs": GPART_LARGE_TASK_CONFIGS,
    },
    "vera": {
        "config_class": VERAConfig,
        "base": VERA_BASE_CONFIG,
        "large": VERA_LARGE_CONFIG,
        "task_configs": VERA_TASK_CONFIGS,
    },
    "unilora": {
        "config_class": UNILORAConfig,
        "base": UNILORA_BASE_CONFIG,
        "large": UNILORA_LARGE_CONFIG,
        "task_configs": UNILORA_TASK_CONFIGS,
        "large_task_configs": UNILORA_LARGE_TASK_CONFIGS,
    },
    "condlora": {
        "config_class": CONDLORAConfig,
        "base": CONDLORA_BASE_CONFIG,
        "large": CONDLORA_LARGE_CONFIG,
        "task_configs": CONDLORA_TASK_CONFIGS,
    },
    "bitfit": {
        "config_class": BITFITConfig,
        "base": BITFIT_BASE_CONFIG,
        "large": BITFIT_LARGE_CONFIG,
        "task_configs": BITFIT_TASK_CONFIGS,
    },
    "head": {
        "config_class": HEADConfig,
        "base": HEAD_BASE_CONFIG,
        "large": HEAD_LARGE_CONFIG,
        "task_configs": HEAD_TASK_CONFIGS,
    },
    "full": {
        "config_class": FULLConfig,
        "base": FULL_BASE_CONFIG,
        "large": FULL_LARGE_CONFIG,
        "task_configs": FULL_TASK_CONFIGS,
    },
}


def get_adapter_config_class(adapter_type: str):
    """Get the configuration class for an adapter type."""
    if adapter_type not in ADAPTER_CONFIG_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return ADAPTER_CONFIG_REGISTRY[adapter_type]["config_class"]


def get_adapter_instance(adapter_type: str, model_size: str = "base"):
    """Get a pre-configured adapter instance."""
    if adapter_type not in ADAPTER_CONFIG_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    registry = ADAPTER_CONFIG_REGISTRY[adapter_type]
    if model_size == "large":
        return registry.get("large", registry["base"])
    return registry["base"]


def get_adapter_task_configs(adapter_type: str) -> dict:
    """Get task-specific configs for an adapter type."""
    if adapter_type not in ADAPTER_CONFIG_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return ADAPTER_CONFIG_REGISTRY[adapter_type]["task_configs"]


__all__ = [
    # Registry
    "ADAPTER_CONFIG_REGISTRY",
    "get_adapter_config_class",
    "get_adapter_instance",
    "get_adapter_task_configs",
    # LoRA
    "LORAConfig",
    "LORA_BASE_CONFIG",
    "LORA_LARGE_CONFIG",
    "LORA_TASK_CONFIGS",
    # GPART
    "GPARTConfig",
    "GPART_BASE_CONFIG",
    "GPART_LARGE_CONFIG",
    "GPART_TASK_CONFIGS",
    "GPART_LARGE_TASK_CONFIGS",
    # VeRA
    "VERAConfig",
    "VERA_BASE_CONFIG",
    "VERA_LARGE_CONFIG",
    "VERA_TASK_CONFIGS",
    # UniLoRA
    "UNILORAConfig",
    "UNILORA_BASE_CONFIG",
    "UNILORA_LARGE_CONFIG",
    "UNILORA_TASK_CONFIGS",
    # CondLoRA
    "CONDLORAConfig",
    "CONDLORA_BASE_CONFIG",
    "CONDLORA_LARGE_CONFIG",
    "CONDLORA_TASK_CONFIGS",
    # BitFit
    "BITFITConfig",
    "BITFIT_BASE_CONFIG",
    "BITFIT_LARGE_CONFIG",
    "BITFIT_TASK_CONFIGS",
    # Head
    "HEADConfig",
    "HEAD_BASE_CONFIG",
    "HEAD_LARGE_CONFIG",
    "HEAD_TASK_CONFIGS",
    # Full
    "FULLConfig",
    "FULL_BASE_CONFIG",
    "FULL_LARGE_CONFIG",
    "FULL_TASK_CONFIGS",
]
