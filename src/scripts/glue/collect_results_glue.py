#!/usr/bin/env python3
"""
Script to collect results from experiment logs and compute statistics.

Usage:
    python collect_results_glue.py experiments/logs/roberta_glue_lora
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from configs.task_configs import TASK_PRIMARY_METRICS


def find_metric_keys(data):
    """Find all metric keys in results data (excluding loss)."""
    metric_keys = []
    for key in data.keys():
        if key.startswith("test_") and not key.endswith("_loss"):
            metric_keys.append(key)
    return metric_keys


def filter_metrics_for_task(task_name, metric_keys):
    """Filter metrics to keep only the primary metric for each task.

    Uses TASK_PRIMARY_METRICS from configs.task_configs as the single
    source of truth for which metric matters per task.
    """
    primary_metric = TASK_PRIMARY_METRICS.get(task_name)
    filtered_metrics = []

    for metric_key in metric_keys:
        clean_metric_name = metric_key.replace("test_", "")
        if primary_metric is None or clean_metric_name == primary_metric:
            filtered_metrics.append(metric_key)

    return filtered_metrics


def collect_results(base_path):
    """
    Collect results from the given base path.

    Args:
        base_path (str): Path to the experiment logs directory

    Returns:
        dict: Dictionary with task names as keys and dict of metric lists as values
              Structure: {task_name: {metric_name: [values]}}
    """
    results = defaultdict(lambda: defaultdict(list))
    seed_counts = defaultdict(int)

    # Convert to Path object for easier handling
    base_path = Path(base_path)

    # Check if base path exists
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return results

    # Iterate through task directories
    for task_dir in base_path.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            seed_count = 0

            # Iterate through seed directories
            for seed_dir in task_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    # Extract seed number and check if it's 0, 1, or 2
                    try:
                        seed_num = int(seed_dir.name.split("_")[1])
                        # if seed_num not in [0, 1, 2]:
                        #     continue
                    except (ValueError, IndexError):
                        continue

                    results_file = seed_dir / "results.json"

                    # Check if results.json exists
                    if results_file.exists():
                        try:
                            with open(results_file, "r") as f:
                                data = json.load(f)

                            # Find all metric keys
                            metric_keys = find_metric_keys(data)
                            # Filter metrics based on task-specific rules
                            filtered_metric_keys = filter_metrics_for_task(
                                task_name, metric_keys
                            )
                            for metric_key in filtered_metric_keys:
                                if metric_key in data:
                                    results[task_name][metric_key].append(
                                        data[metric_key]
                                    )
                            seed_count += 1
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Could not read {results_file}: {e}")

            seed_counts[task_name] = seed_count

    return results, seed_counts


def compute_stats(results):
    """
    Compute median and standard deviation for each task and metric.

    Args:
        results (dict): Dictionary with task names as keys and dict of metric lists as values

    Returns:
        dict: Dictionary with task names as keys and dict of metric stats as values
              Structure: {task_name: {metric_name: (median, std)}}
    """
    stats = defaultdict(dict)

    for task, metrics in results.items():
        for metric_name, values in metrics.items():
            if values:  # Only compute stats if we have values
                median = np.median(values)
                std = np.std(values) if len(values) > 1 else 0.0
                stats[task][metric_name] = (median, std)
            else:
                stats[task][metric_name] = (0.0, 0.0)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect results from experiment logs")
    parser.add_argument("path", help="Path to the experiment logs directory")
    args = parser.parse_args()

    # Collect results
    results, seed_counts = collect_results(args.path)

    # Compute statistics
    stats = compute_stats(results)

    # Print task and seed info
    print("Task Information:")
    for task in sorted(seed_counts.keys()):
        print(f"  {task}: {seed_counts[task]} seeds")
    print()

    # Define the required column order
    column_order = ["cola", "sst2", "mrpc", "stsb", "qnli", "rte"]

    # Extract values for each column and calculate averages
    column_values = {}
    column_stats = {}
    all_metric_values = []

    for task in column_order:
        if task in stats and stats[task]:
            # Get the first (and should be only) metric for this task
            metrics = stats[task]
            metric_name = list(metrics.keys())[0]  # Should be the filtered metric
            median, std = metrics[metric_name]
            median_100 = median * 100
            std_100 = std * 100
            column_values[task] = median_100
            column_stats[task] = (median_100, std_100)
            all_metric_values.append(median_100)
        else:
            column_values[task] = 0.0
            column_stats[task] = (0.0, 0.0)

    # Calculate overall average
    if all_metric_values:
        overall_avg = sum(all_metric_values) / len(all_metric_values)
    else:
        overall_avg = 0.0

    # Print CSV format with column names
    csv_header = ",".join(column_order + ["avg"])
    print(csv_header)
    csv_values = [
        f"{column_stats[task][0]:.1f} ({column_stats[task][1]:.1f})"
        for task in column_order
    ] + [f"{overall_avg:.1f}"]
    print(",".join(csv_values))


if __name__ == "__main__":
    main()
