"""
MLflow utilities for GLUE fine-tuning experiments.

This module provides context managers and helper functions for MLflow
experiment tracking, simplifying the integration with the training script.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

# =============================================================================
# MLflow Constants
# =============================================================================

MLFLOW_CONFIG_ARTIFACT_NAME = "configurations.json"
MLFLOW_MAIN_CONFIG_ARTIFACT_NAME = "main_config.json"

# Check MLflow availability
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


def check_mlflow_available(logger=None):
    """
    Check MLflow availability and log warning if not available.

    Args:
        logger: Optional logger for warning message

    Returns:
        bool: Whether MLflow is available
    """
    if not MLFLOW_AVAILABLE:
        if logger:
            logger.warning(
                "MLflow not available. Install with 'pip install mlflow' for experiment tracking."
            )
        return False
    return True


@contextmanager
def mlflow_experiment_context(
    experiment_name: str,
    run_name: str,
    config: Optional[Any] = None,
    log_params: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for MLflow experiment tracking.

    Automatically starts and ends an MLflow run, with optional config logging.

    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
        config: Optional configuration object to log as artifact
        log_params: Optional dictionary of parameters to log

    Yields:
        The MLflow run object (or None if MLflow not available)

    Example:
        ```python
        with mlflow_experiment_context("glue_lora", "run_001", config) as run:
            # Your training code
            mlflow.log_metric("accuracy", 0.95)
        ```
    """
    run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)

            # Log additional parameters
            if log_params:
                mlflow.log_params(log_params)

            # Log configuration as artifact
            if config:
                config_dict = _config_to_dict(config)
                mlflow.log_dict(config_dict, MLFLOW_MAIN_CONFIG_ARTIFACT_NAME)

        except Exception as e:
            if hasattr(logger, "warning") if "logger" in locals() else True:
                print(f"Failed to initialize MLflow: {e}")
            run = None

    try:
        yield run
    finally:
        if MLFLOW_AVAILABLE and run:
            try:
                mlflow.end_run()
            except Exception:
                pass


@contextmanager
def nested_mlflow_run(
    run_name: str,
    parent_run: Optional[Any] = None,
):
    """
    Context manager for nested MLflow runs (for task-level tracking).

    Args:
        run_name: Name of the nested run
        parent_run: Parent run (used for nesting)

    Yields:
        The nested MLflow run object (or None if MLflow not available)
    """
    run = None
    if MLFLOW_AVAILABLE and parent_run:
        try:
            run = mlflow.start_run(run_name=run_name, nested=True)
        except Exception:
            run = None

    try:
        yield run
    finally:
        if MLFLOW_AVAILABLE and run:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert a configuration object to a dictionary for MLflow logging.

    Handles both the new dataclass-based configs and the old YACS configs.

    Args:
        config: Configuration object

    Returns:
        Dictionary representation of the config
    """
    config_dict = {}

    if hasattr(config, "adapter"):
        # New dataclass-based config
        if hasattr(config.adapter, "to_dict"):
            config_dict["adapter"] = config.adapter.to_dict()
        else:
            config_dict["adapter"] = {
                k: getattr(config.adapter, k)
                for k in dir(config.adapter)
                if not k.startswith("_")
            }

    if hasattr(config, "training"):
        config_dict["training"] = {
            "batch_size": getattr(config.training, "batch_size", 32),
            "max_seq_length": getattr(config.training, "max_seq_length", 512),
            "weight_decay": getattr(config.training, "weight_decay", 0.1),
            "lr": getattr(config.training, "lr", 4e-4),
            "head_lr": getattr(config.training, "head_lr", 1e-3),
        }

    if hasattr(config, "task_metadata"):
        config_dict["task_metadata"] = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in config.task_metadata.items()
        }

    return config_dict


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],
    prefix: str = "",
):
    """
    Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric name -> value
        prefix: Optional prefix for metric names
    """
    if not MLFLOW_AVAILABLE:
        return

    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            full_name = f"{prefix}_{name}" if prefix else name
            try:
                mlflow.log_metric(full_name, value)
            except Exception:
                pass


def log_params_to_mlflow(
    params: Dict[str, Any],
    prefix: str = "",
):
    """
    Log parameters to MLflow.

    Args:
        params: Dictionary of parameter name -> value
        prefix: Optional prefix for parameter names
    """
    if not MLFLOW_AVAILABLE:
        return

    for name, value in params.items():
        full_name = f"{prefix}_{name}" if prefix else name
        try:
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(full_name, value)
            elif isinstance(value, (list, dict)):
                mlflow.log_param(full_name, str(value))
        except Exception:
            pass


def log_config_to_mlflow(
    adapter_config: Dict[str, Any],
    task_config: Dict[str, Any],
    training_args: Dict[str, Any],
):
    """
    Log complete configuration as MLflow artifact.

    Args:
        adapter_config: Adapter configuration dictionary
        task_config: Task configuration dictionary
        training_args: Training arguments dictionary
    """
    if not MLFLOW_AVAILABLE:
        return

    config_data = {
        "adapter_config": adapter_config,
        "task_config": task_config,
        "training_args": training_args,
    }

    try:
        mlflow.log_dict(config_data, MLFLOW_CONFIG_ARTIFACT_NAME)
    except Exception:
        pass


def get_experiment_run_name(
    adapter_type: str,
    timestamp: Optional[str] = None,
    isometric: bool = True,
    experiment_type: str = "glue",
) -> tuple:
    """
    Generate MLflow experiment and run names.

    Args:
        adapter_type: Type of adapter
        timestamp: Optional timestamp string (defaults to current time)
        isometric: Whether using isometric initialization (for GPart)
        experiment_type: Type of experiment (glue, math, etc.)

    Returns:
        Tuple of (experiment_name, run_name)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Add isometric suffix for GPart if not isometric
    isometric_suffix = ""
    if adapter_type == "gpart" and not isometric:
        isometric_suffix = "_isoFalse"

    experiment_name = f"{experiment_type}_{adapter_type}{isometric_suffix}"
    run_name = (
        f"{experiment_type}_experiment_{adapter_type}{isometric_suffix}_{timestamp}"
    )

    return experiment_name, run_name


def log_final_results_to_mlflow(
    all_results: Dict[str, Dict[str, Any]],
):
    """
    Log final aggregated results to MLflow.

    Args:
        all_results: Dictionary of task_name -> metrics
    """
    if not MLFLOW_AVAILABLE:
        return

    for task_name, task_results in all_results.items():
        for metric_name, metric_values in task_results.items():
            if isinstance(metric_values, dict) and "mean" in metric_values:
                try:
                    mlflow.log_metric(
                        f"{task_name}_{metric_name}_mean", metric_values["mean"]
                    )
                    mlflow.log_metric(
                        f"{task_name}_{metric_name}_std", metric_values["std"]
                    )
                except Exception:
                    pass
