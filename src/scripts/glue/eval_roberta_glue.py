"""
Evaluation-only Script for RoBERTa-base/large on Individual GLUE Tasks

Loads a previously trained adapter checkpoint and evaluates it on the task's
validation split, writing ``results.json`` in the same format as the training
script so that ``collect_results_glue.py`` can pick it up.

This is useful when training completed (the adapter was saved) but the final
test-set evaluation step did not run or crashed before saving results.

The seed folder paths are given directly so the script works regardless of
the directory naming convention.

Usage:
    python src/scripts/glue/eval_roberta_glue.py \
        --adapter_type unilora \
        --task cola \
        --model_size base \
        --seed_dirs \
            experiments/outputs/roberta_base_glue_unilora/cola/seed_0 \
            experiments/outputs/roberta_base_glue_unilora/cola/seed_2
"""

import argparse
import json
import os
import warnings

from configs.config import (
    ALLOWED_ADAPTERS,
    ALLOWED_TASKS,
    load_config,
)
from utils.adapter_utils import logger
from utils.model_utils import load_model, load_pretrained_adapter
from utils.trainer import (
    ADAPTER_SUBDIR,
    BEST_MODEL_SUBDIR,
    RESULTS_FILENAME,
    evaluate_test,
    prepare_datasets,
)
from utils.training_config import RunTaskConfig

warnings.filterwarnings("ignore")


# =============================================================================
# CLI + main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved RoBERTa GLUE adapter without retraining"
    )
    parser.add_argument(
        "--adapter_type",
        required=True,
        help="Type of adapter to use.",
        choices=ALLOWED_ADAPTERS,
    )
    parser.add_argument("--config_file", type=str, help="Custom config file path.")
    parser.add_argument(
        "--model_size",
        choices=["base", "large"],
        default="base",
        help="RoBERTa model size to use.",
    )
    parser.add_argument("--base_model", help="HF model name (overrides model_size).")
    parser.add_argument(
        "--task",
        required=True,
        choices=ALLOWED_TASKS,
        help="GLUE task to evaluate.",
    )
    parser.add_argument(
        "--seed_dirs",
        nargs="+",
        required=True,
        help="Paths to seed folders (each containing adapter/best_model).",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def evaluate_single(task_name, seed_dir, args):
    """Evaluate one (task, seed_dir) pair and save results.json."""

    # Resolve base model name (same logic as ExperimentArgs.from_args)
    base_model = args.base_model
    if base_model is None:
        base_model = f"roberta-{args.model_size}"

    # Load task-specific config
    task_specific_config = load_config(
        adapter_type=args.adapter_type,
        task_name=task_name,
        config_file=args.config_file,
        model_size=args.model_size,
        opts=args.opts,
    )

    # Build RunTaskConfig (same factory as the training script).
    # output_root and seed are not used for path construction here since the
    # seed directory is given directly, but RunTaskConfig still needs them for
    # dataset/eval metadata. We pass the seed_dir as output_root and seed=0
    # as placeholders — they are not used in the eval-only path.
    task_config = RunTaskConfig.from_configs(
        task_name=task_name,
        base_model_name=base_model,
        output_root=seed_dir,
        seed=0,
        adapter_type=args.adapter_type,
        config=task_specific_config,
        main_seed=0 if args.adapter_type == "gpart" else None,
    )

    # Locate the saved adapter checkpoint directly inside the seed folder
    adapter_path = os.path.join(seed_dir, ADAPTER_SUBDIR, BEST_MODEL_SUBDIR)

    if not os.path.isdir(adapter_path):
        logger.error(
            f"Adapter checkpoint not found at {adapter_path} — skipping {seed_dir}."
        )
        return None

    logger.info(f"Evaluating {task_name.upper()} from {adapter_path}")

    # Load tokenizer and prepare test dataset (only test_loader is used)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(task_config.base_model_name)
    _, _, test_loader = prepare_datasets(task_config, tokenizer)

    # Load base model
    problem_type = (
        "regression"
        if task_config.is_regression_task()
        else "single_label_classification"
    )
    model = load_model(
        task_config.base_model_name,
        task_config.get_num_labels(),
        problem_type,
    )

    # Load the saved adapter
    model = load_pretrained_adapter(model, adapter_path, logger)

    # Evaluate
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_metrics = evaluate_test(
        model, test_loader, task_config.task_name, device, logger
    )
    logger.info(f"Task {task_name.upper()} ({seed_dir}) test metrics: {test_metrics}")

    # Save results.json in the seed directory
    results_path = os.path.join(seed_dir, RESULTS_FILENAME)
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return test_metrics


def main():
    args = parse_args()

    logger.info(
        f"Starting evaluation-only run: adapter={args.adapter_type}, "
        f"task={args.task}, seed_dirs={args.seed_dirs}"
    )

    all_results = {}
    for seed_dir in args.seed_dirs:
        try:
            metrics = evaluate_single(args.task, seed_dir, args)
            if metrics is not None:
                all_results[seed_dir] = metrics
        except Exception:
            logger.exception(f"Evaluation failed for {seed_dir}.")

    logger.info("Evaluation-only run complete.")
    logger.info(f"Results: {json.dumps(all_results, indent=2)}")


if __name__ == "__main__":
    main()
