#!/usr/bin/env python3
"""
Script to collect results from multiple seed runs and compute statistics.

Reads results.json files from the hierarchical directory layout:
    vit-{model_size}/{adapter_info}-epoch{N}/{dataset}/seed_{seed}/results.json

Usage:
    python collect_results_ViT.py experiments/outputs/vit-base
    python collect_results_ViT.py experiments/outputs/vit-base/unilora-r4-d23040-epoch20
    python collect_results_ViT.py experiments/outputs/vit-base/unilora-r4-d23040-epoch20/cifar10
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Collect results from multiple seed runs and compute statistics"
    )
    parser.add_argument("path", help="Path to the experiment directory")
    args = parser.parse_args()

    folder_path = args.path

    if not os.path.isdir(folder_path):
        print(f"Error: Input directory '{folder_path}' does not exist")
        return

    # key = group key (config + dataset, seed-normalized), value = list of accuracies
    grouped_results = defaultdict(list)

    # Recursively walk the directory tree to find results.json files
    for dirpath, _dirnames, filenames in os.walk(folder_path):
        if "results.json" not in filenames:
            continue

        file_path = os.path.join(dirpath, "results.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            acc = data.get("eval_accuracy", None)
            if acc is None:
                acc = data.get("test_accuracy", None)
            if acc is not None:
                # Build group key from relative path, normalizing seed numbers
                rel_path = os.path.relpath(dirpath, folder_path)
                group_key = re.sub(r"seed_\d+", "seed_N", rel_path)
                grouped_results[group_key].append(acc)
        except Exception as e:
            print(f"  {file_path} : {e}")

    mean_acc_list = []

    # Also store per-group stats for LaTeX output
    # method_name -> dataset -> (mean, std)
    method_dataset_stats = defaultdict(dict)

    for group_key, accs in sorted(grouped_results.items()):
        mean_acc = statistics.mean(accs)
        mean_acc_list.append(mean_acc)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{group_key}: mean={mean_acc:.4f}, std={std_acc:.4f}, n={len(accs)}")

        # Parse group_key for LaTeX row: {adapter_info}-epoch{N}/{dataset}/seed_N
        parts = group_key.split("/")
        if len(parts) >= 2:
            adapter_epoch = parts[0]
            dataset = parts[1]

            # Remove -epoch{N} suffix to get adapter_info
            adapter_info = re.sub(r"-epoch\d+$", "", adapter_epoch)

            # Extract method name
            if adapter_info.startswith("gpart"):
                method_name = "GPart"
            elif adapter_info.startswith("unilora"):
                method_name = "UniLoRA"
            elif adapter_info.startswith("lora"):
                method_name = "LoRA"
            else:
                method_name = adapter_info

            method_dataset_stats[method_name][dataset] = (mean_acc * 100, std_acc * 100)

    if mean_acc_list:
        print(f"\nAvg. accuracy: {statistics.mean(mean_acc_list):.4f}")
    else:
        print("No results found.")
        return

    # --- LaTeX row output ---
    DATASET_ORDER = [
        "oxfordpets",
        "standfordcars",
        "cifar10",
        "dtd",
        "eurosat",
        "fgvc",
        "resisc45",
        "cifar100",
    ]

    # Extract model name from path
    if "vit-large" in folder_path:
        model_name = "ViT-L"
    else:
        model_name = "ViT-B"

    print("\n% LaTeX table rows:")
    for method_name, dataset_stats in sorted(method_dataset_stats.items()):
        cells = [model_name, method_name, "0"]  # Model, Method, # Params (fill manually)

        dataset_values = []
        for ds in DATASET_ORDER:
            if ds in dataset_stats:
                mean_val, std_val = dataset_stats[ds]
                cells.append(f"{mean_val:.2f}" + "$_{\\pm " + f"{std_val:.2f}" + "}$")
                dataset_values.append(mean_val)
            else:
                cells.append("—")

        # Compute average over available datasets
        if dataset_values:
            avg = sum(dataset_values) / len(dataset_values)
            cells.append(f"{avg:.2f}")
        else:
            cells.append("—")

        print(" & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
