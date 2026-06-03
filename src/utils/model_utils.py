"""
Model utilities for GLUE fine-tuning experiments.

This module provides functions for model creation, parameter logging,
and adapter application.
"""

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification


def load_model(
    base_model_name: str,
    num_labels: int,
    problem_type: str = "single_label_classification",
    ignore_mismatched_sizes: bool = True,
) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained model for sequence classification.

    Args:
        base_model_name: HuggingFace model name or path
        num_labels: Number of output labels
        problem_type: Type of problem (regression, single_label_classification, multi_label_classification)
        ignore_mismatched_sizes: Whether to ignore size mismatches when loading

    Returns:
        AutoModelForSequenceClassification instance
    """
    return AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )


def apply_adapter(
    model,
    adapter_type: str,
    adapter_config: Dict[str, Any],
    target_modules: List[str],
    get_peft_config_fn,
    logger=None,
):
    """
    Apply an adapter to a model.

    Handles three cases:
    1. Head-only: Only train classifier layer
    2. BitFit: Only train bias terms
    3. Full: All parameters trainable
    4. PEFT adapters: Use get_peft_model

    Args:
        model: The model to apply adapter to
        adapter_type: Type of adapter
        adapter_config: Adapter configuration dictionary
        target_modules: Target modules for PEFT adapters
        get_peft_config_fn: Function to get PEFT config
        logger: Optional logger for info messages

    Returns:
        Model with adapter applied
    """
    from peft import PeftModel, get_peft_model

    if adapter_type == "head":
        # Head-only: Only train classifier layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        if logger:
            logger.info("Head-only fine-tuning enabled.")

    elif adapter_type == "bitfit":
        # BitFit: Only train bias terms
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        if logger:
            logger.info("BitFit (bias-only fine-tuning) enabled.")

    elif adapter_type == "full":
        # Full fine-tuning: All parameters trainable
        if logger:
            logger.info("Full fine-tuning enabled (no adapters).")

    else:
        # PEFT adapters (LoRA, VeRA, GPART, etc.)
        peft_cfg = get_peft_config_fn(adapter_type, adapter_config, target_modules)
        if isinstance(peft_cfg, tuple):
            for cfg in peft_cfg:
                model = get_peft_model(model, cfg)
        else:
            model = get_peft_model(model, peft_cfg)

    return model


def load_pretrained_adapter(
    model,
    adapter_path: str,
    logger=None,
):
    """
    Load a pre-trained adapter into a model.

    Args:
        model: Base model to load adapter into
        adapter_path: Path to pre-trained adapter
        logger: Optional logger

    Returns:
        Model with pre-trained adapter loaded
    """
    from peft import PeftModel

    if logger:
        logger.info(f"Loading pre-trained adapter from {adapter_path}")

    return PeftModel.from_pretrained(model, adapter_path)


def get_parameter_counts(model) -> Tuple[int, int]:
    """
    Get trainable and total parameter counts.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, all_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params


def get_classifier_param_count(model) -> int:
    """
    Get the number of parameters in the classifier layer.

    Args:
        model: PyTorch model with classifier attribute

    Returns:
        Number of classifier parameters
    """
    return sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)


def log_detailed_parameters(model, adapter_type: str, logger) -> Dict[str, Any]:
    """
    Log detailed parameter information.

    Args:
        model: PyTorch model
        adapter_type: Type of adapter
        logger: Logger instance

    Returns:
        Dictionary with parameter statistics
    """
    # Handle both PEFT models and base models
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, all_params = model.get_nb_trainable_parameters()
    else:
        trainable_params, all_params = get_parameter_counts(model)

    trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0

    # Calculate classifier parameters
    classifier_params = get_classifier_param_count(model)
    adapter_params = trainable_params - classifier_params

    # Get base model parameters (non-trainable)
    base_model_params = all_params - trainable_params

    # Adapter parameters as percentage of total
    adapter_total_percentage = (
        100 * adapter_params / all_params if all_params > 0 else 0
    )

    # Memory estimation (4 bytes per parameter for fp32)
    memory_mb = trainable_params * 4 / (1024 * 1024)

    # Log parameter breakdown
    logger.info("=" * 60)
    logger.info(f"PARAMETER BREAKDOWN - {adapter_type.upper()} ADAPTER")
    logger.info("=" * 60)
    logger.info(f"Total Parameters     : {all_params:>12,}")
    logger.info(
        f"Trainable Parameters : {trainable_params:>12,} ({trainable_percentage:>6.2f}%)"
    )
    logger.info(
        f"Base Model Parameters: {base_model_params:>12,} ({100 - trainable_percentage:>6.2f}%)"
    )
    logger.info(
        f"Adapter Parameters   : {adapter_params:>12,} ({adapter_total_percentage:>6.4f}% of total)"
    )
    logger.info("-" * 60)
    logger.info("TRAINABLE PARAMETER BREAKDOWN:")
    logger.info(
        f"  Adapter Parameters : {adapter_params:>12,} ({100 * adapter_params / trainable_params:>6.2f}% of trainable)"
    )
    logger.info(
        f"  Classifier         : {classifier_params:>12,} ({100 * classifier_params / trainable_params:>6.2f}% of trainable)"
    )
    logger.info(f"Estimated Memory   : {memory_mb:>12.2f} MB")
    logger.info("=" * 60)

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "classifier_params": classifier_params,
        "adapter_params": adapter_params,
        "base_model_params": base_model_params,
        "trainable_percentage": trainable_percentage,
        "adapter_percentage": adapter_total_percentage,
        "memory_mb": memory_mb,
    }


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Move a batch of tensors to the specified device.

    Args:
        batch: Dictionary of tensors
        device: Target device ("cuda" or "cpu")

    Returns:
        Batch with all tensors on the target device
    """
    return {k: v.to(device) for k, v in batch.items()}


def get_problem_type(is_regression: bool) -> str:
    """
    Get the problem type string for model configuration.

    Args:
        is_regression: Whether this is a regression task

    Returns:
        Problem type string
    """
    return "regression" if is_regression else "single_label_classification"
