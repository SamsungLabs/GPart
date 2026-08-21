#!/usr/bin/env python3
"""
Standalone evaluation script for ViT adapters trained with finetune_ViT.py.

Supports two usage modes:

Mode A — evaluate a specific seed directory (compatible with eval_vit_base_unilora.sh):
    python src/scripts/vision/eval_ViT.py experiments/outputs/vit-base/gpart-d72000-epoch20/cifar10/seed_0

Mode B — evaluate an adapter config directory on specified datasets:
    python src/scripts/vision/eval_ViT.py experiments/outputs/vit-base/gpart-d72000-epoch20 --datasets cifar10 standfordcars

In Mode B, the script iterates over the requested datasets and all seed_*
subdirectories found under each dataset, evaluating each adapter on the test set.
"""

import argparse
import glob
import json
import os
import random

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import peft
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate a trained ViT PEFT adapter on one or more datasets."
)
parser.add_argument(
    "adapter_path",
    type=str,
    help=(
        "Path to either a seed directory (e.g. .../cifar10/seed_0) or an adapter "
        "config directory (e.g. .../gpart-d72000-epoch20)."
    ),
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Datasets to evaluate when adapter_path is a config directory. "
        "If omitted, the dataset is inferred from the path (Mode A)."
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Evaluate only this seed number (default: all seeds found).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Evaluation batch size (default: 64 for base, 32 for large).",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    default=True,
    help="Use fp16 for evaluation (default: True).",
)
parser.add_argument(
    "--no_fp16",
    dest="fp16",
    action="store_false",
    help="Disable fp16 for evaluation.",
)
parser.add_argument(
    "--output_suffix",
    type=str,
    default="_eval",
    help=(
        "Suffix appended to the results filename (default: '_eval', producing "
        "results_eval.json). Pass an empty string to overwrite results.json."
    ),
)
parser.add_argument(
    "--assignment_backend",
    type=str,
    default=None,
    choices=["legacy_streaming", "implicit_stateless_v1"],
    help=(
        "Override the GPart assignment_backend when loading a GPart adapter. "
        "Use 'legacy_streaming' (torch.randint-based) or 'implicit_stateless_v1' "
        "(stateless SplitMix64 hash). Only applies to GPart adapters; ignored for "
        "other PEFT types. WARNING: changing the backend from what was used during "
        "training will produce different group assignments and scales, likely "
        "degrading accuracy — the saved theta_d was optimized for the original backend."
    ),
)

args = parser.parse_args()


KNOWN_DATASETS = [
    "oxfordpets",
    "fgvc",
    "standfordcars",
    "cifar10",
    "cifar100",
    "dtd",
    "flowers102",
    "eurosat",
    "resisc45",
]


# ---------------------------------------------------------------------------
# Dataset loading (mirrors finetune_ViT.py)
# ---------------------------------------------------------------------------

def load_dataset_splits(dataset_name, seed=42):
    """Load train/val/test splits for the given dataset.

    Returns (train_ds, val_ds, test_ds, is_torchvision_dataset).
    Only the test split is actually needed for evaluation, but train_ds is
    used to derive label mappings.
    """
    is_torchvision_dataset = False

    if dataset_name == "cifar10":
        train_val_ds = load_dataset("cifar10", split="train")
        train_valid_split = train_val_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = train_valid_split["train"]
        val_ds = train_valid_split["test"]
        test_ds = load_dataset("cifar10", split="test")

    elif dataset_name == "cifar100":
        train_val_ds = load_dataset("cifar100", split="train")
        train_valid_split = train_val_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = train_valid_split["train"]
        val_ds = train_valid_split["test"]
        test_ds = load_dataset("cifar100", split="test")

    elif dataset_name == "flowers102":
        train_ds = load_dataset("oxford_flowers102", split="train")
        val_ds = load_dataset("oxford_flowers102", split="validation")
        test_ds = load_dataset("oxford_flowers102", split="test")

    elif dataset_name == "resisc45":
        train_ds = load_dataset("timm/resisc45", split="train")
        val_ds = load_dataset("timm/resisc45", split="validation")
        test_ds = load_dataset("timm/resisc45", split="test")

    elif dataset_name == "oxfordpets":
        train_val_ds = load_dataset("timm/oxford-iiit-pet", split="train")
        train_valid_split = train_val_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = train_valid_split["train"]
        val_ds = train_valid_split["test"]
        test_ds = load_dataset("timm/oxford-iiit-pet", split="test")

    elif dataset_name == "standfordcars":
        train_val_ds = load_dataset("tanganke/stanford_cars", split="train")
        train_valid_split = train_val_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = train_valid_split["train"]
        val_ds = train_valid_split["test"]
        test_ds = load_dataset("tanganke/stanford_cars", split="test")

    elif dataset_name == "fgvc":
        from torchvision.datasets import FGVCAircraft

        train_ds = FGVCAircraft(
            root="data", split="train", annotation_level="variant", download=True
        )
        val_ds = FGVCAircraft(
            root="data", split="val", annotation_level="variant", download=True
        )
        test_ds = FGVCAircraft(
            root="data", split="test", annotation_level="variant", download=True
        )
        is_torchvision_dataset = True

    elif dataset_name == "dtd":
        ds = load_dataset("imagefolder", data_dir="data/dtd/images", split="train")
        ds = ds.shuffle(seed=seed)
        train_val_test = ds.train_test_split(test_size=0.28, seed=seed)
        val_test = train_val_test["test"].train_test_split(test_size=0.715, seed=seed)
        train_ds = train_val_test["train"]
        val_ds = val_test["train"]
        test_ds = val_test["test"]

    elif dataset_name == "eurosat":
        train_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
        val_ds = load_dataset("blanchon/EuroSAT_RGB", split="validation")
        test_ds = load_dataset("blanchon/EuroSAT_RGB", split="test")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_ds, val_ds, test_ds, is_torchvision_dataset


def derive_label_maps(train_ds, is_torchvision_dataset):
    """Derive label2id / id2label from the training dataset."""
    if is_torchvision_dataset:
        labels = train_ds.classes
        label2id = train_ds.class_to_idx
        id2label = {i: label for label, i in label2id.items()}
        return labels, label2id, id2label, None

    if "label" in train_ds.features:
        hf_label_column = "label"
    elif "fine_label" in train_ds.features:
        hf_label_column = "fine_label"
    else:
        raise ValueError(
            f"Cannot find label column in dataset: {train_ds.features.keys()}"
        )

    if isinstance(train_ds.features[hf_label_column], ClassLabel):
        labels = train_ds.features[hf_label_column].names
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
    else:
        unique_labels = sorted(set(train_ds[hf_label_column]))
        label2id = {str(i): i for i in unique_labels}
        id2label = {i: str(i) for i in unique_labels}

    return labels, label2id, id2label, hf_label_column


def build_transforms(image_processor):
    """Build val/test transforms from the image processor."""
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    test_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )
    return test_transforms


def prepare_test_dataset(
    test_ds, train_ds, is_torchvision_dataset, hf_label_column, test_transforms
):
    """Apply transforms to the test dataset and return the processed dataset."""
    if not is_torchvision_dataset:
        if "image" in train_ds.features:
            image_column = "image"
        elif "img" in train_ds.features:
            image_column = "img"
        else:
            raise ValueError(
                f"Can't find image column in dataset: {train_ds.features.keys()}"
            )

        def preprocess_test(example_batch):
            example_batch["pixel_values"] = [
                test_transforms(image.convert("RGB"))
                for image in example_batch[image_column]
            ]
            example_batch["labels"] = example_batch[hf_label_column]
            return example_batch

        test_ds.set_transform(preprocess_test)
        return test_ds

    else:
        class TorchvisionFGVCDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                image, label = self.base_dataset[idx]
                image = image.convert("RGB")
                pixel_values = self.transform(image)
                return {
                    "pixel_values": pixel_values,
                    "labels": label,
                }

        return TorchvisionFGVCDataset(test_ds, test_transforms)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate_adapter(
    adapter_dir,
    dataset_name,
    batch_size=None,
    fp16=True,
    output_suffix="_eval",
    assignment_backend_override=None,
):
    """Load and evaluate a single adapter on the given dataset's test set.

    Args:
        assignment_backend_override: If not None and the adapter is a GPart adapter,
            override the assignment_backend field in the loaded config before
            constructing the model. This regenerates group indices/scales using the
            new backend. WARNING: changing the backend from what was used during
            training will produce different group assignments and scales.

    Returns the evaluation results dict.
    """
    # --- Read adapter config ---
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    peft_type = adapter_config.get("peft_type", "UNKNOWN")
    print(f"  PEFT type: {peft_type}")
    print(f"  Base model: {base_model_name}")

    modules_to_save = adapter_config.get("modules_to_save") or []
    if "classifier" not in modules_to_save:
        print(
            "  WARNING: this adapter does not contain a saved classifier. "
            "Evaluation will use a newly initialized classification head."
        )

    # --- Determine if assignment_backend override applies ---
    use_override = (
        assignment_backend_override is not None
        and peft_type == "GPART"
    )
    if use_override:
        saved_backend = adapter_config.get(
            "assignment_backend", "legacy_streaming"
        )
        if saved_backend != assignment_backend_override:
            print(
                f"  ⚠️  Overriding assignment_backend: "
                f"{saved_backend} -> {assignment_backend_override}"
            )
            print(
                f"  ⚠️  WARNING: theta_d was trained with '{saved_backend}'. "
                f"Using a different backend produces different group assignments "
                f"and scales — accuracy will likely degrade."
            )
        else:
            print(
                f"  assignment_backend override matches saved config "
                f"({saved_backend}), no change."
            )
            use_override = False  # no actual change needed


    # --- Determine model size for default batch size ---
    if "vit-large" in base_model_name:
        model_size = "large"
    else:
        model_size = "base"
    if batch_size is None:
        batch_size = 32 if model_size == "large" else 64

    # --- Load dataset ---
    print(f"  Loading dataset: {dataset_name}")
    train_ds, val_ds, test_ds, is_torchvision_dataset = load_dataset_splits(dataset_name)

    # --- Derive label maps ---
    labels, label2id, id2label, hf_label_column = derive_label_maps(
        train_ds, is_torchvision_dataset
    )

    # --- Load image processor (prefer saved preprocessor, fall back to base model) ---
    preprocessor_path = os.path.join(adapter_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_path):
        image_processor = AutoImageProcessor.from_pretrained(adapter_dir)
    else:
        image_processor = AutoImageProcessor.from_pretrained(base_model_name)

    # --- Build transforms and prepare test dataset ---
    test_transforms = build_transforms(image_processor)
    test_ds = prepare_test_dataset(
        test_ds, train_ds, is_torchvision_dataset, hf_label_column, test_transforms
    )

    # --- Load base model ---
    print(f"  Loading base model...")
    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # --- Load adapter ---
    print(f"  Loading adapter from: {adapter_dir}")
    if use_override:
        # Load config, override assignment_backend, build model, then load weights
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        peft_config.assignment_backend = assignment_backend_override
        print(f"  Building GPart model with assignment_backend={assignment_backend_override}")
        model = get_peft_model(model, peft_config)

        # Load the saved adapter weights (theta_d) into the freshly built model.
        # gpart_indices and gpart_global_scales are non-persistent buffers that
        # were regenerated during get_peft_model using the overridden backend.
        adapter_weights = load_peft_weights(adapter_dir)
        load_result = set_peft_model_state_dict(model, adapter_weights)
        if load_result.unexpected_keys:
            print(f"  Unexpected keys: {load_result.unexpected_keys}")
    else:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()


    # --- Evaluate ---
    eval_args = TrainingArguments(
        output_dir=adapter_dir,
        remove_unused_columns=False,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        report_to="none",
        label_names=["labels"],
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    print(f"  Running evaluation on test set...")
    results = trainer.evaluate(test_ds)

    print(f"  Results: {results}")

    # --- Save results ---
    results_filename = "results.json"
    if output_suffix:
        results_filename = f"results{output_suffix}.json"
    results_path = os.path.join(adapter_dir, results_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # Clean up to free memory before the next adapter
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_evaluation_targets(adapter_path, datasets, seed_filter):
    """Resolve adapter_path into a list of (adapter_dir, dataset_name) tuples.

    Mode A: adapter_path is a seed directory (e.g. .../cifar10/seed_0)
    Mode B: adapter_path is a config directory (e.g. .../gpart-d72000-epoch20)
    """
    targets = []

    # Check if adapter_path itself contains an adapter_config.json (Mode A)
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        # Infer dataset from parent directory
        parent_dir = os.path.basename(os.path.dirname(adapter_path.rstrip("/")))
        if parent_dir in KNOWN_DATASETS:
            dataset_name = parent_dir
        else:
            raise ValueError(
                f"Could not infer dataset name from path: {adapter_path}. "
                f"Parent directory '{parent_dir}' is not a known dataset. "
                f"Use --datasets to specify."
            )
        targets.append((adapter_path.rstrip("/"), dataset_name))
        return targets

    # Mode B: adapter_path is a config directory
    if datasets is None:
        # Auto-discover datasets from subdirectories
        for entry in sorted(os.listdir(adapter_path)):
            full_path = os.path.join(adapter_path, entry)
            if os.path.isdir(full_path) and entry in KNOWN_DATASETS:
                datasets = datasets or []
                if entry not in datasets:
                    datasets.append(entry)

    if not datasets:
        raise ValueError(
            f"No datasets specified and could not auto-discover any in: {adapter_path}"
        )

    for dataset_name in datasets:
        dataset_dir = os.path.join(adapter_path, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"  ⚠️  Dataset directory not found, skipping: {dataset_dir}")
            continue

        # Find all seed directories
        if seed_filter is not None:
            seed_dirs = [os.path.join(dataset_dir, f"seed_{seed_filter}")]
            seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
        else:
            seed_dirs = sorted(
                glob.glob(os.path.join(dataset_dir, "seed_*"))
            )
            seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]

        for seed_dir in seed_dirs:
            if os.path.exists(os.path.join(seed_dir, "adapter_config.json")):
                targets.append((seed_dir, dataset_name))
            else:
                print(f"  ⚠️  No adapter_config.json in, skipping: {seed_dir}")

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"PEFT version: {peft.__version__}")
    print(f"Adapter path: {args.adapter_path}")
    if args.datasets:
        print(f"Datasets: {args.datasets}")
    if args.seed is not None:
        print(f"Seed filter: {args.seed}")
    print()

    targets = resolve_evaluation_targets(args.adapter_path, args.datasets, args.seed)

    if not targets:
        print("No adapters found to evaluate.")
        return

    print(f"Found {len(targets)} adapter(s) to evaluate:\n")
    for adapter_dir, dataset_name in targets:
        print(f"  {dataset_name}: {adapter_dir}")
    print()

    all_results = {}
    for i, (adapter_dir, dataset_name) in enumerate(targets):
        print(f"========================================")
        print(f"Evaluating [{i + 1}/{len(targets)}]")
        print(f"  Dataset: {dataset_name}")
        print(f"  Adapter: {adapter_dir}")
        print(f"========================================")

        try:
            results = evaluate_adapter(
                adapter_dir=adapter_dir,
                dataset_name=dataset_name,
                batch_size=args.batch_size,
                fp16=args.fp16,
                output_suffix=args.output_suffix,
                assignment_backend_override=args.assignment_backend,
            )

            all_results[f"{dataset_name}/{os.path.basename(adapter_dir)}"] = results
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

        print()

    # Print summary
    print("========================================")
    print("Evaluation Summary")
    print("========================================")
    for key, results in all_results.items():
        acc = results.get("eval_accuracy", "N/A")
        print(f"  {key}: accuracy = {acc}")
    print("========================================")


if __name__ == "__main__":
    main()
