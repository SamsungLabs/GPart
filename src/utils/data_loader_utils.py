"""
Data loading utilities for GLUE fine-tuning experiments.

This module provides functions for dataset loading, tokenization,
and DataLoader creation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

# =============================================================================
# Data Loading Constants
# =============================================================================

# Dataset splitting for validation
MIN_VAL_SAMPLES = 300  # Minimum number of validation samples
VAL_FRACTION_MIN = 0.15  # Minimum validation fraction (15%)
VAL_FRACTION_MAX = 0.20  # Maximum validation fraction (20%)

# DataLoader configuration
DATALOADER_NUM_WORKERS = 8
VAL_BATCH_SIZE_MULTIPLIER = 2  # Validation batch size = train_batch_size * multiplier
PIN_MEMORY = True


def split_train_val(
    train_dataset: Dataset,
    seed: int,
    is_regression: bool = False,
    label_column: str = "label",
) -> Tuple[Dataset, Dataset]:
    """
    Split training dataset into train and validation sets.

    Uses adaptive validation size: at least 300 samples, between 15-20% of data.

    Args:
        train_dataset: The training dataset to split
        seed: Random seed for reproducibility
        is_regression: Whether this is a regression task (no stratification)
        label_column: Name of the label column for stratification

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_size = len(train_dataset)

    # Adaptive val size: at least MIN_VAL_SAMPLES samples, max 20%
    val_fraction = max(VAL_FRACTION_MIN, MIN_VAL_SAMPLES / train_size)
    val_fraction = min(val_fraction, VAL_FRACTION_MAX)  # Cap at 20%

    if is_regression:
        # For regression, don't use stratification
        split = train_dataset.train_test_split(
            test_size=val_fraction,
            seed=seed,
        )
    else:
        # For classification, use stratification
        split = train_dataset.train_test_split(
            test_size=val_fraction,
            seed=seed,
            stratify_by_column=label_column,
        )

    return split["train"], split["test"]


def create_tokenize_fn(
    tokenizer: AutoTokenizer,
    text_keys: List[Optional[str]],
    max_seq_length: int,
) -> Callable:
    """
    Create a tokenization function for dataset mapping.

    Args:
        tokenizer: The tokenizer to use
        text_keys: List of text column keys (e.g., ["sentence", None] or ["sentence1", "sentence2"])
        max_seq_length: Maximum sequence length

    Returns:
        Tokenization function suitable for dataset.map()
    """

    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Handle single or multiple text inputs
        if len(text_keys) == 1 or text_keys[1] is None:
            # Single input (e.g., SST-2, CoLA)
            text = examples[text_keys[0]]
            return tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,  # Dynamic padding with collator
            )
        else:
            # Paired input (e.g., MRPC, STS-B, QNLI, RTE)
            text1 = examples[text_keys[0]]
            text2 = examples[text_keys[1]]
            return tokenizer(
                text1,
                text2,
                truncation=True,
                max_length=max_seq_length,
                padding=False,  # Dynamic padding with collator
            )

    return tokenize_fn


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    text_keys: List[Optional[str]],
    max_seq_length: int,
    columns_to_remove: Optional[List[str]] = None,
) -> Dataset:
    """
    Tokenize a dataset.

    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        text_keys: Text column keys
        max_seq_length: Maximum sequence length
        columns_to_remove: Columns to remove after tokenization

    Returns:
        Tokenized dataset
    """
    tokenize_fn = create_tokenize_fn(tokenizer, text_keys, max_seq_length)

    mapped = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=columns_to_remove,
    )

    return mapped


def prepare_dataset_for_training(
    dataset: Dataset,
    label_column: str = "label",
) -> Dataset:
    """
    Prepare a tokenized dataset for training.

    - Renames 'label' column to 'labels' (expected by transformers)
    - Sets format to torch tensors

    Args:
        dataset: Tokenized dataset
        label_column: Name of the label column

    Returns:
        Prepared dataset
    """
    dataset = dataset.rename_column(label_column, "labels")
    dataset.set_format("torch")
    return dataset


def get_columns_to_remove(
    dataset: Dataset,
    keep_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Get list of columns to remove from dataset.

    Args:
        dataset: Dataset to analyze
        keep_columns: Columns to keep (default: label, input_ids, attention_mask, token_type_ids)

    Returns:
        List of column names to remove
    """
    if keep_columns is None:
        keep_columns = ["label", "input_ids", "attention_mask", "token_type_ids"]

    return [col for col in dataset.column_names if col not in keep_columns]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    collator: DataCollatorWithPadding,
    shuffle: bool = False,
    num_workers: int = DATALOADER_NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        collator: Data collator for padding
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    collator: DataCollatorWithPadding,
) -> DataLoader:
    """Create a training DataLoader with shuffling."""
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        collator=collator,
        shuffle=True,
    )


def create_eval_dataloader(
    dataset: Dataset,
    batch_size: int,
    collator: DataCollatorWithPadding,
) -> DataLoader:
    """Create an evaluation DataLoader without shuffling."""
    # Use larger batch size for evaluation (faster)
    eval_batch_size = batch_size * VAL_BATCH_SIZE_MULTIPLIER
    return create_dataloader(
        dataset,
        batch_size=eval_batch_size,
        collator=collator,
        shuffle=False,
    )


def prepare_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    test_dataset: Dataset,
    batch_size: int,
    collator: DataCollatorWithPadding,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Create all DataLoaders for training, validation, and testing.

    When val_dataset is None (model_selection="last"), no val_loader is
    created and the full training set is used.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (None if no validation split)
        test_dataset: Test dataset (for final evaluation)
        batch_size: Training batch size
        collator: Data collator

    Returns:
        Tuple of (train_loader, val_loader_or_None, test_loader)
    """
    train_loader = create_train_dataloader(train_dataset, batch_size, collator)
    val_loader = (
        create_eval_dataloader(val_dataset, batch_size, collator)
        if val_dataset is not None
        else None
    )
    test_loader = create_eval_dataloader(test_dataset, batch_size, collator)

    return train_loader, val_loader, test_loader
