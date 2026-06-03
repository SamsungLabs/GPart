"""
Task metadata configuration for GLUE fine-tuning experiments.

This module contains task-specific metadata such as dataset names,
text keys, number of labels, and evaluation metrics. It is the single
source of truth for all task metadata — other modules derive their
constants from here.
"""

from typing import Dict, List, Optional, Tuple

from configs.base_config import TaskMetadata

# Default task metadata for GLUE tasks
# Extracted from base.yaml task_metadata section
GLUE_TASK_METADATA: Dict[str, TaskMetadata] = {
    "cola": TaskMetadata(
        dataset=("nyu-mll/glue", "cola"),
        num_labels=2,
        text_keys=["sentence", None],
        metric_fn="matthews_correlation",
        split="validation",
    ),
    "sst2": TaskMetadata(
        dataset=("nyu-mll/glue", "sst2"),
        num_labels=2,
        text_keys=["sentence", None],
        metric_fn="accuracy",
        split="validation",
    ),
    "mrpc": TaskMetadata(
        dataset=("nyu-mll/glue", "mrpc"),
        num_labels=2,
        text_keys=["sentence1", "sentence2"],
        metric_fn="accuracy",
        split="validation",
    ),
    "stsb": TaskMetadata(
        dataset=("nyu-mll/glue", "stsb"),
        num_labels=1,  # Regression task
        text_keys=["sentence1", "sentence2"],
        metric_fn="pearson",
        split="validation",
    ),
    "qnli": TaskMetadata(
        dataset=("nyu-mll/glue", "qnli"),
        num_labels=2,
        text_keys=["question", "sentence"],
        metric_fn="accuracy",
        split="validation",
    ),
    "rte": TaskMetadata(
        dataset=("nyu-mll/glue", "rte"),
        num_labels=2,
        text_keys=["sentence1", "sentence2"],
        metric_fn="accuracy",
        split="validation",
    ),
}

# Primary metrics for each task — derived from GLUE_TASK_METADATA so there is
# a single source of truth.  Used for early stopping and best model selection.
TASK_PRIMARY_METRICS: Dict[str, str] = {
    name: meta.metric_fn for name, meta in GLUE_TASK_METADATA.items()
}

# Regression tasks (require special handling) — derived from metadata
REGRESSION_TASKS: List[str] = [
    name for name, meta in GLUE_TASK_METADATA.items() if meta.num_labels == 1
]


def get_task_metadata(task_name: str) -> TaskMetadata:
    """Get task metadata by name, falling back to defaults."""
    if task_name in GLUE_TASK_METADATA:
        return GLUE_TASK_METADATA[task_name]

    # Return default metadata for unknown tasks
    return TaskMetadata(
        dataset=("nyu-mll/glue", task_name),
        num_labels=2,
        text_keys=["sentence", None],
        metric_fn="accuracy",
        split="validation",
    )


def get_all_task_metadata() -> Dict[str, TaskMetadata]:
    """Get all task metadata."""
    return GLUE_TASK_METADATA.copy()


__all__ = [
    "GLUE_TASK_METADATA",
    "TASK_PRIMARY_METRICS",
    "REGRESSION_TASKS",
    "get_task_metadata",
    "get_all_task_metadata",
]
