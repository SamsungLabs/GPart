"""
Unified Fine-tuning & Evaluation Script for RoBERTa-base/large on Individual GLUE Tasks
Supports various adapters with their respective configurations.

Pipeline:
  1. For each GLUE task, fine-tune roberta-base with the specified adapter (sequence classification).
  2. Evaluate on the validation split and report per-task metrics.
  3. Compute and print the GLUE average (using primary metric per task).
"""

import argparse
import json
import os
import warnings
from datetime import datetime

# Import configuration management
from configs.config import (
    ALLOWED_ADAPTERS,
    ALLOWED_TASKS,
    get_adapter_config,
    load_config,
)

# Import refactored utilities
from utils.adapter_utils import get_peft_config, logger, print_summary
from utils.mlflow_utils import (
    is_mlflow_available,
    log_final_results_to_mlflow,
    mlflow_experiment_context,
    nested_mlflow_run,
)
from utils.trainer import get_gpart_suffixes, run_task
from utils.training_config import ExperimentArgs, RunTaskConfig

warnings.filterwarnings("ignore")


# =============================================================================
# CLI + main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified fine-tuning of RoBERTa-base on GLUE"
    )
    parser.add_argument(
        "--adapter_type",
        required=True,
        help="Type of adapter to use.",
        choices=ALLOWED_ADAPTERS,
    )
    parser.add_argument("--load_adapter", type=str, help="Path to pre-trained adapter.")
    parser.add_argument("--config_file", type=str, help="Custom config file path.")
    parser.add_argument(
        "--model_size",
        choices=["base", "large"],
        default="base",
        help="RoBERTa model size to use.",
    )
    parser.add_argument("--base_model", help="HF model name (overrides model_size).")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALLOWED_TASKS,
        help="GLUE tasks.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/outputs",
        help="Root output directory.",
    )
    parser.add_argument(
        "--compute_params_only", action="store_true", help="Only compute parameters."
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create structured experiment args (handles base_model and output_dir logic)
    exp_args = ExperimentArgs.from_args(args)

    # Load config with model size information for proper config selection
    config = load_config(
        adapter_type=exp_args.adapter_type,
        config_file=exp_args.config_file,
        load_adapter=exp_args.load_adapter,
        model_size=exp_args.model_size,
        opts=exp_args.opts,
    )

    # Only create output directory if not in compute_params_only mode
    # For GPart, we'll create the directory with isometric suffix in run_task
    if not exp_args.compute_params_only and exp_args.adapter_type != "gpart":
        os.makedirs(exp_args.output_dir, exist_ok=True)

    # Generate timestamp for unique run naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Build MLflow experiment/run names
    if exp_args.adapter_type == "gpart":
        adapter_config = get_adapter_config(config, exp_args.adapter_type)
        iso_suffix, grp_suffix = get_gpart_suffixes(adapter_config)
        experiment_name = f"glue_{exp_args.adapter_type}{iso_suffix}{grp_suffix}"
        run_name = f"glue_experiment_{exp_args.adapter_type}{iso_suffix}{grp_suffix}_{timestamp}"
    else:
        experiment_name = f"glue_{exp_args.adapter_type}"
        run_name = f"glue_experiment_{exp_args.adapter_type}_{timestamp}"

    # Run with MLflow context
    with mlflow_experiment_context(
        experiment_name,
        run_name,
        config=config,
        log_params={
            "adapter_type": exp_args.adapter_type,
            "base_model": exp_args.base_model,
            "seed": exp_args.seed,
            "tasks": ", ".join(exp_args.tasks),
            "output_dir": exp_args.output_dir,
        },
    ) as mlflow_run:
        if is_mlflow_available() and mlflow_run and exp_args.config_file:
            import mlflow

            mlflow.log_param("config_file", exp_args.config_file)

        logger.info(f"Starting experiment with adapter: {exp_args.adapter_type}")
        all_results = {}

        for task_name in exp_args.tasks:
            with nested_mlflow_run(f"task_{task_name}", mlflow_run) as task_mlflow_run:
                # Refresh task-specific config
                task_specific_config = load_config(
                    adapter_type=exp_args.adapter_type,
                    task_name=task_name,
                    config_file=exp_args.config_file,
                    load_adapter=exp_args.load_adapter,
                    model_size=exp_args.model_size,
                    opts=exp_args.opts,
                )

                # Create RunTaskConfig from the config system
                task_config = RunTaskConfig.from_configs(
                    task_name=task_name,
                    base_model_name=exp_args.base_model,
                    output_root=exp_args.output_dir,
                    seed=exp_args.seed,
                    adapter_type=exp_args.adapter_type,
                    config=task_specific_config,
                    compute_params_only=exp_args.compute_params_only,
                    main_seed=exp_args.seed,
                    load_adapter=exp_args.load_adapter,
                    task_mlflow_run=task_mlflow_run,
                )

                # Run task using the refactored trainer
                try:
                    agg = run_task(task_config, get_peft_config, logger)
                    all_results[task_name] = agg
                except Exception:
                    logger.exception(f"Task '{task_name}' failed.")

        # Print summary and save results
        if not exp_args.compute_params_only:
            print_summary(
                all_results, exp_args.tasks, exp_args.adapter_type.upper(), config
            )

            # Build results file path (with GPart suffixes)
            if exp_args.adapter_type == "gpart":
                adapter_config = get_adapter_config(config, exp_args.adapter_type)
                iso_suffix, grp_suffix = get_gpart_suffixes(adapter_config)
                results_dir = exp_args.output_dir + iso_suffix + grp_suffix
            else:
                results_dir = exp_args.output_dir

            out_path = os.path.join(
                results_dir, f"glue_{exp_args.adapter_type}_results.json"
            )
            with open(out_path, "w") as fh:
                json.dump(all_results, fh, indent=2)
            logger.info(f"Full results saved to: {out_path}")

            # Log final results to MLflow
            if is_mlflow_available() and mlflow_run:
                log_final_results_to_mlflow(all_results)
        else:
            logger.info("Parameter mode: Detailed summary printed to console.")


if __name__ == "__main__":
    main()
