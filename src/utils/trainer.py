"""
Training utilities for GLUE fine-tuning experiments.

This module provides the core training loop functionality,
extracted from the original monolithic run_task function.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from configs.task_configs import TASK_PRIMARY_METRICS
from utils import data_loader_utils
from utils.mlflow_utils import (
    log_config_to_mlflow,
    log_final_results_to_mlflow,
    log_metrics_to_mlflow,
    log_params_to_mlflow,
)
from utils.model_utils import (
    apply_adapter,
    load_model,
    load_pretrained_adapter,
    log_detailed_parameters,
    move_batch_to_device,
)
from utils.training_config import RunTaskConfig

# =============================================================================
# Training & Output Constants
# =============================================================================

# Output directory constants
DEFAULT_OUTPUT_ROOT = "experiments/outputs"
ADAPTER_SUBDIR = "adapter"
BEST_MODEL_SUBDIR = "best_model"
RESULTS_FILENAME = "results.json"


def get_gpart_suffixes(adapter_config: dict) -> tuple:
    """Build isometric and grouping_strategy suffixes for GPart adapters.

    Args:
        adapter_config: Adapter configuration dictionary

    Returns:
        Tuple of (isometric_suffix, grouping_suffix)
    """
    isometric_suffix = ""
    if "isometric" in adapter_config and adapter_config["isometric"] == False:
        isometric_suffix = "_iso{}".format(str(adapter_config["isometric"]).lower())

    grouping_suffix = ""
    if (
        "grouping_strategy" in adapter_config
        and adapter_config["grouping_strategy"] != "random"
    ):
        grouping_suffix = f"_{adapter_config['grouping_strategy']}"

    return isometric_suffix, grouping_suffix


def get_output_dir(
    output_root: str,
    task_name: str,
    seed: int,
    adapter_type: str,
    adapter_config: Dict[str, Any],
) -> str:
    """
    Get the output directory path, handling GPart isometric and grouping_strategy suffixes.

    Args:
        output_root: Root output directory
        task_name: Task name
        seed: Random seed
        adapter_type: Type of adapter
        adapter_config: Adapter configuration

    Returns:
        Output directory path
    """
    if adapter_type == "gpart":
        iso_suffix, grp_suffix = get_gpart_suffixes(adapter_config)
        output_root = output_root + iso_suffix + grp_suffix

    return os.path.join(output_root, task_name, f"seed_{seed}")


def prepare_datasets(
    config: RunTaskConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Prepare train, validation, and test datasets.

    When model_selection="last", no validation split is created — the full
    training set is used for training, and val_loader is None.

    Args:
        config: RunTaskConfig with all parameters
        tokenizer: Tokenizer for text processing

    Returns:
        Tuple of (train_loader, val_loader_or_None, test_loader)
    """
    from datasets import load_dataset

    # Load raw dataset
    ds_path, ds_cfg = config.get_dataset_info()
    raw = load_dataset(ds_path, ds_cfg)

    # Get evaluation split
    eval_split = config.get_eval_split()

    # Use validation split as test set
    if eval_split in raw:
        test_dataset = raw[eval_split]
    else:
        test_dataset = raw["train"]

    # Get columns to remove
    columns_to_remove = data_loader_utils.get_columns_to_remove(raw["train"])
    text_keys = config.get_text_keys()

    # When model_selection="last", use full training data (no val split)
    if config.model_selection == "last":
        train_dataset = raw["train"]
        val_dataset = None
    else:
        # Split training data for validation (best-model selection)
        is_regression = config.is_regression_task()
        train_dataset, val_dataset = data_loader_utils.split_train_val(
            raw["train"],
            config.seed,
            is_regression=is_regression,
        )

    # Tokenize datasets
    tokenized_train = data_loader_utils.tokenize_dataset(
        train_dataset,
        tokenizer,
        text_keys,
        config.max_seq_length,
        columns_to_remove,
    )
    tokenized_val = (
        data_loader_utils.tokenize_dataset(
            val_dataset,
            tokenizer,
            text_keys,
            config.max_seq_length,
            columns_to_remove,
        )
        if val_dataset is not None
        else None
    )
    tokenized_test = data_loader_utils.tokenize_dataset(
        test_dataset,
        tokenizer,
        text_keys,
        config.max_seq_length,
        columns_to_remove,
    )

    # Prepare for training
    tokenized_train = data_loader_utils.prepare_dataset_for_training(tokenized_train)
    if tokenized_val is not None:
        tokenized_val = data_loader_utils.prepare_dataset_for_training(tokenized_val)
    tokenized_test = data_loader_utils.prepare_dataset_for_training(tokenized_test)

    # Create collator
    collator = DataCollatorWithPadding(tokenizer)

    # Create DataLoaders
    train_loader, val_loader, test_loader = data_loader_utils.prepare_data_loaders(
        tokenized_train,
        tokenized_val,
        tokenized_test,
        config.batch_size,
        collator,
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    train_steps = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        train_steps += 1

    return total_loss / train_steps if train_steps > 0 else 0


def _evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    task_name: str,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Core evaluation loop shared by evaluate_epoch and evaluate_test.

    Args:
        model: Model to evaluate
        eval_loader: Evaluation DataLoader
        task_name: Task name for metric selection
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    metric = evaluate.load("glue", task_name)
    total_loss = 0
    eval_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            eval_steps += 1

            if task_name == "stsb":
                preds = logits.squeeze(-1).detach().cpu().numpy().astype(float)
                refs = batch["labels"].detach().cpu().numpy().astype(float)
                metric.add_batch(predictions=preds, references=refs)
            else:
                preds = logits.argmax(dim=-1)
                metric.add_batch(predictions=preds, references=batch["labels"])

    avg_loss = total_loss / eval_steps if eval_steps > 0 else 0
    metrics = metric.compute()

    return avg_loss, metrics


def evaluate_epoch(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    task_name: str,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate for one epoch.

    Args:
        model: Model to evaluate
        eval_loader: Evaluation DataLoader
        task_name: Task name for metric selection
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    return _evaluate(model, eval_loader, task_name, device)


def run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: RunTaskConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    adapter_output_dir: str,
    tokenizer: AutoTokenizer,
    logger,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Run the full training loop.

    When model_selection="best" (default), validates each epoch and keeps
    the model with the best validation score.

    When model_selection="last", skips validation entirely and returns the
    model from the last epoch. This uses 100% of the training data.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (None if model_selection="last")
        config: RunTaskConfig
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        adapter_output_dir: Output directory for checkpoints
        tokenizer: Tokenizer for saving
        logger: Logger instance

    Returns:
        Tuple of (model, metrics dict)
    """
    primary_metric = TASK_PRIMARY_METRICS.get(config.task_name, "accuracy")

    if config.model_selection == "last":
        # ── Last-epoch mode: no validation, use all training data ──
        for epoch in range(config.epochs):
            logger.info(
                f"[{config.task_name.upper()}] Epoch {epoch+1}/{config.epochs} - Training"
            )
            avg_train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            logger.info(f"Epoch {epoch+1}/{config.epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")

        # Save last-epoch model
        if not config.compute_params_only:
            last_ckpt_path = os.path.join(adapter_output_dir, "last_model")
            os.makedirs(last_ckpt_path, exist_ok=True)
            model.save_pretrained(last_ckpt_path)
            tokenizer.save_pretrained(last_ckpt_path)
            logger.info(f"  Last-epoch model saved to: {last_ckpt_path}")

        logger.info(
            f"Using last-epoch model (epoch {config.epochs}), "
            f"no validation was performed"
        )
        return model, {primary_metric: 0.0}

    else:
        # ── Best-model mode: validate each epoch, keep best ──
        best_score = -float("inf")
        best_model_state = None
        best_epoch = -1

        for epoch in range(config.epochs):
            # Training
            logger.info(
                f"[{config.task_name.upper()}] Epoch {epoch+1}/{config.epochs} - Training"
            )
            avg_train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device
            )

            # Validation
            logger.info(
                f"[{config.task_name.upper()}] Epoch {epoch+1}/{config.epochs} - Validation"
            )
            avg_val_loss, val_metrics = evaluate_epoch(
                model, val_loader, config.task_name, device
            )
            current_score = val_metrics[primary_metric]

            logger.info(f"Epoch {epoch+1}/{config.epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  Val {primary_metric}: {current_score:.4f}")

            # Track best model
            if current_score > best_score:
                best_score = current_score
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch + 1
                logger.info(f"  New best {primary_metric}: {best_score:.4f}")

                # Save best model
                if not config.compute_params_only:
                    best_ckpt_path = os.path.join(
                        adapter_output_dir, BEST_MODEL_SUBDIR
                    )
                    os.makedirs(best_ckpt_path, exist_ok=True)
                    model.save_pretrained(best_ckpt_path)
                    tokenizer.save_pretrained(best_ckpt_path)
                    logger.info(f"  Best model saved to: {best_ckpt_path}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(
                f"Loaded best model (epoch {best_epoch}) with "
                f"{primary_metric}: {best_score:.4f}"
            )

        return model, {primary_metric: best_score}


def evaluate_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    task_name: str,
    device: str,
    logger,
) -> Dict[str, float]:
    """
    Evaluate on test set.

    Args:
        model: Model to evaluate
        test_loader: Test DataLoader
        task_name: Task name
        device: Device
        logger: Logger

    Returns:
        Dictionary of test metrics with 'test_' prefix
    """
    logger.info(f"[{task_name.upper()}] Final Test Evaluation")

    avg_loss, metrics = _evaluate(model, test_loader, task_name, device)

    # Format metrics with test_ prefix
    clean_metrics = {f"test_{k}": round(v, 4) for k, v in metrics.items()}
    clean_metrics["test_loss"] = round(avg_loss, 4)

    return clean_metrics


def _log_section(logger, title: str, items: Dict[str, Any], width: int = 60):
    """Log a config section with a title and key-value pairs."""
    logger.info(f"  ── {title} {'─' * (width - 6 - len(title))}")
    for key, value in items.items():
        logger.info(f"  {key:<28}: {value}")
    logger.info("")


def log_run_config(config: RunTaskConfig, logger):
    """
    Log the core, adapter, and training configuration at training start.

    This is called once at the beginning of run_task() to give a clear
    overview of the experiment setup. Task-specific config is logged
    separately via log_task_config() right before training each task.

    Training fields are derived from TrainingConfig via cached_property,
    so adding a new field to TrainingConfig automatically makes it
    available here.
    """
    from dataclasses import asdict

    width = 60
    logger.info("═" * width)
    logger.info(
        f"  CONFIGURATION — TASK: {config.task_name.upper()} | "
        f"ADAPTER: {config.adapter_type.upper()}"
    )
    logger.info("═" * width)

    # Core parameters
    _log_section(logger, "Core", {
        "task_name": config.task_name,
        "base_model": config.base_model_name,
        "seed": config.seed,
        "output_root": config.output_root,
        "adapter_type": config.adapter_type,
    })

    # Adapter configuration
    if config.adapter_config:
        _log_section(logger, "Adapter", config.adapter_config)
    else:
        logger.info("  ── Adapter ──────────────────────────────")
        logger.info("  No adapter configuration found")
        logger.info("")

    # Training parameters — derived from TrainingConfig via cached_property
    _log_section(logger, "Training", asdict(config.config.training))

    logger.info("═" * width)
    logger.info("")


def log_task_config(config: RunTaskConfig, logger):
    """
    Log task-specific configuration (dataset, metrics, etc.).

    Called right before training starts for each individual task,
    so the task details appear contextually when they're relevant.
    """
    if not config.task_config:
        return

    width = 60
    logger.info("─" * width)
    logger.info(f"  TASK CONFIG — {config.task_name.upper()}")
    logger.info("─" * width)
    for key, value in config.task_config.items():
        logger.info(f"  {key:<22}: {value}")
    logger.info("─" * width)
    logger.info("")


def run_task(
    config: RunTaskConfig,
    get_peft_config_fn,
    logger,
) -> Dict[str, float]:
    """
    Run complete fine-tuning and evaluation for a single task.

    This is the refactored version of the original run_task function,
    now using helper functions from this module.

    Args:
        config: RunTaskConfig with all parameters
        get_peft_config_fn: Function to get PEFT config
        logger: Logger instance

    Returns:
        Dictionary of test metrics
    """
    from .adapter_utils import build_optimizer_and_scheduler

    # Print configurations
    log_run_config(config, logger)

    # Set seed
    from transformers import set_seed

    set_seed(config.seed)

    # Pass main_seed to adapter config for GPart reproducibility
    if config.adapter_type == "gpart" and config.main_seed is not None:
        config.adapter_config["main_seed"] = config.main_seed

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Prepare datasets
    train_loader, val_loader, test_loader = prepare_datasets(config, tokenizer)

    # Load model
    problem_type = (
        "regression" if config.is_regression_task() else "single_label_classification"
    )
    model = load_model(
        config.base_model_name,
        config.get_num_labels(),
        problem_type,
    )

    # Load pre-trained adapter if specified
    if config.load_adapter:
        model = load_pretrained_adapter(model, config.load_adapter, logger)

    # Apply adapter
    model = apply_adapter(
        model,
        config.adapter_type,
        config.adapter_config,
        config.target_modules,
        get_peft_config_fn,
        logger,
    )

    # Log parameters
    param_stats = log_detailed_parameters(model, config.adapter_type, logger)

    # Handle compute_params_only mode
    if config.compute_params_only:
        return {
            "total_params": param_stats["trainable_params"],
            "classifier_params": param_stats["classifier_params"],
            "non_classifier_params": param_stats["trainable_params"]
            - param_stats["classifier_params"],
        }

    # Prepare output directory
    task_output_dir = get_output_dir(
        config.output_root,
        config.task_name,
        config.seed,
        config.adapter_type,
        config.adapter_config,
    )
    adapter_output_dir = os.path.join(task_output_dir, ADAPTER_SUBDIR)
    os.makedirs(adapter_output_dir, exist_ok=True)

    # Training setup
    total_steps = config.get_total_steps(len(train_loader.dataset))
    warmup_steps = config.get_warmup_steps(total_steps)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        adapter_lr=config.base_lr,
        head_lr=config.head_lr,
        weight_decay=config.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        adapter_type=config.adapter_type,
    )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Log task-specific config right before training starts
    log_task_config(config, logger)

    # Run training
    model, _ = run_training_loop(
        model,
        train_loader,
        val_loader,
        config,
        optimizer,
        scheduler,
        device,
        adapter_output_dir,
        tokenizer,
        logger,
    )

    # Evaluate on test set
    test_metrics = evaluate_test(model, test_loader, config.task_name, device, logger)
    logger.info(f"Task {config.task_name.upper()} final test metrics: {test_metrics}")

    # Log to MLflow
    if config.task_mlflow_run is not None:
        try:
            log_metrics_to_mlflow(test_metrics)
            log_params_to_mlflow({"trainable_params": param_stats["trainable_params"]})
            log_params_to_mlflow(config.adapter_config, prefix="adapter")
            log_config_to_mlflow(
                config.adapter_config,
                config.task_config,
                {
                    "num_train_epochs": config.epochs,
                    "per_device_train_batch_size": config.batch_size,
                    "max_seq_length": config.max_seq_length,
                    "weight_decay": config.weight_decay,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    # Save results
    if not config.compute_params_only:
        results_path = os.path.join(task_output_dir, RESULTS_FILENAME)
        with open(results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

    return test_metrics
