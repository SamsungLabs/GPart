"""
Utility modules for GLUE fine-tuning experiments.

This package provides various utilities for training, data loading,
model handling, and experiment tracking.
"""

from configs.task_configs import REGRESSION_TASKS, TASK_PRIMARY_METRICS

# Re-export constants from their new homes
from .data_loader_utils import (
    DATALOADER_NUM_WORKERS,
    MIN_VAL_SAMPLES,
    PIN_MEMORY,
    VAL_BATCH_SIZE_MULTIPLIER,
    VAL_FRACTION_MAX,
    VAL_FRACTION_MIN,
    create_dataloader,
    create_eval_dataloader,
    create_tokenize_fn,
    create_train_dataloader,
    get_columns_to_remove,
    prepare_data_loaders,
    prepare_dataset_for_training,
    split_train_val,
    tokenize_dataset,
)
from .mlflow_utils import (
    MLFLOW_CONFIG_ARTIFACT_NAME,
    MLFLOW_MAIN_CONFIG_ARTIFACT_NAME,
    check_mlflow_available,
    get_experiment_run_name,
    is_mlflow_available,
    log_config_to_mlflow,
    log_final_results_to_mlflow,
    log_metrics_to_mlflow,
    log_params_to_mlflow,
    mlflow_experiment_context,
    nested_mlflow_run,
)
from .model_utils import (
    apply_adapter,
    get_classifier_param_count,
    get_parameter_counts,
    get_problem_type,
    load_model,
    load_pretrained_adapter,
    log_detailed_parameters,
    move_batch_to_device,
)
from .trainer import (
    ADAPTER_SUBDIR,
    BEST_MODEL_SUBDIR,
    DEFAULT_OUTPUT_ROOT,
    RESULTS_FILENAME,
    evaluate_epoch,
    evaluate_test,
    get_gpart_suffixes,
    get_output_dir,
    prepare_datasets,
    run_task,
    run_training_loop,
    train_epoch,
)
from .training_config import ExperimentArgs, RunTaskConfig

__all__ = [
    # Constants
    "DATALOADER_NUM_WORKERS",
    "MIN_VAL_SAMPLES",
    "VAL_FRACTION_MIN",
    "VAL_FRACTION_MAX",
    "VAL_BATCH_SIZE_MULTIPLIER",
    "PIN_MEMORY",
    "TASK_PRIMARY_METRICS",
    "REGRESSION_TASKS",
    "MLFLOW_CONFIG_ARTIFACT_NAME",
    "MLFLOW_MAIN_CONFIG_ARTIFACT_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "ADAPTER_SUBDIR",
    "BEST_MODEL_SUBDIR",
    "RESULTS_FILENAME",
    # Config classes
    "RunTaskConfig",
    "ExperimentArgs",
    # Data loading
    "split_train_val",
    "create_tokenize_fn",
    "tokenize_dataset",
    "prepare_dataset_for_training",
    "get_columns_to_remove",
    "create_dataloader",
    "create_train_dataloader",
    "create_eval_dataloader",
    "prepare_data_loaders",
    # Model utilities
    "load_model",
    "apply_adapter",
    "load_pretrained_adapter",
    "get_parameter_counts",
    "get_classifier_param_count",
    "log_detailed_parameters",
    "move_batch_to_device",
    "get_problem_type",
    # MLflow utilities
    "is_mlflow_available",
    "check_mlflow_available",
    "mlflow_experiment_context",
    "nested_mlflow_run",
    "log_metrics_to_mlflow",
    "log_params_to_mlflow",
    "log_config_to_mlflow",
    "get_experiment_run_name",
    "log_final_results_to_mlflow",
    # Trainer
    "get_output_dir",
    "prepare_datasets",
    "train_epoch",
    "evaluate_epoch",
    "run_training_loop",
    "evaluate_test",
    "run_task",
]
