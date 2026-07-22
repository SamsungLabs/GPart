import logging
import typing as T

from torch.optim import AdamW
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from configs.task_configs import TASK_PRIMARY_METRICS
from peft import LoraConfig, UniLoRAConfig, VeraConfig
from peft.tuners.condlora import CondLoraConfig
from peft.tuners.gpart import GPartConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Suppress HTTP and verbose logging from HuggingFace libraries and HTTP clients
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# PEFT Configuration Factory
# =============================================================================
def get_peft_config(
    adapter_type: str,
    adapter_config: T.Dict[str, T.Any],
    target_modules: T.List[str],
) -> T.Any:
    """Factory to create PEFT configuration objects."""
    if adapter_type in ["head", "bitfit"]:
        return None

    CONFIG_CLASSES = {
        "lora": (LoraConfig, {"lora_alpha": "alpha", "lora_dropout": "dropout"}),
        "vera": (VeraConfig, {"vera_dropout": "dropout"}),
        "condlora": (
            CondLoraConfig,
            {"lora_alpha": "alpha", "lora_dropout": "dropout"},
        ),
        "unilora": (UniLoRAConfig, {"unilora_dropout": "dropout"}),
        "gpart": (GPartConfig, {"gpart_dropout": "dropout"}),
    }

    config_cls, arg_map = CONFIG_CLASSES.get(
        adapter_type,
        (CondLoraConfig, {"lora_alpha": "alpha", "lora_dropout": "dropout"}),
    )

    # Map common keys from adapter_config to specific PEFT config arguments
    kwargs = {
        k: adapter_config.get(v, adapter_config.get(k)) for k, v in arg_map.items()
    }

    # Add other common adapter parameters
    for k in [
        "r",
        "target_modules",
        "d_initial",
        "projection_prng_key",
        "theta_d_length",
        "init_theta_d_bound",
    ]:
        if k in adapter_config:
            kwargs[k] = adapter_config[k]

    if adapter_type == "gpart":
        for k in [
            "d",
            "init_bound",
            "isometric",
            "grouping_strategy",
            "bias",
            "assignment_backend",
        ]:
            if k in adapter_config:
                kwargs[k] = adapter_config[k]
        # Always use the main seed for GPart proj_seed to ensure reproducibility
        kwargs["proj_seed"] = adapter_config.get(
            "main_seed", 0
        )  # Default to 0 if not provided
    # Add target modules if not already in kwargs
    if "target_modules" not in kwargs:
        kwargs["target_modules"] = target_modules

    # Add bias parameter if not already provided (for non-gpart adapters)
    if "bias" not in kwargs:
        kwargs["bias"] = "none"

    return config_cls(
        **kwargs,
        modules_to_save=["classifier"],
        **({"save_projection": True} if "vera" in adapter_type else {}),
        **(
            {
                "fan_in_fan_out": False,
                "init_lora_weights": True,
                "use_x": "none",
                "lora_x_scaling": 0.0,
            }
            if adapter_type == "condlora"
            else {}
        ),
    )


# =============================================================================
# Print summary
# =============================================================================
def print_summary(
    all_results: T.Dict[str, T.Any],
    tasks: T.List[str],
    method_name: str,
    config: T.Any = None,
):
    """Print GLUE results summary."""
    # TASK_PRIMARY_METRICS is derived from GLUE_TASK_METADATA (single source of truth),
    # so no config-based override is needed.
    primary_keys = TASK_PRIMARY_METRICS

    logger.info(f"GLUE Results Summary - {method_name}")
    print(f"\n{'=' * 70}")
    print(f"  GLUE Results  (RoBERTa-base + {method_name})  —  single seed results")
    print("=" * 70)
    print(f"  {'Task':<10} {'Primary Metric':<26} {'Score':>8}")
    print(f"  {'-' * 44}")

    scores = []
    for task in tasks:
        key = primary_keys.get(task)
        if task in all_results:
            # Try to find the metric key in various formats
            score = None
            # Check direct key match
            if key in all_results[task]:
                score = all_results[task][key]
            # Check with 'eval_' prefix (common in Trainer results)
            elif f"eval_{key}" in all_results[task]:
                score = all_results[task][f"eval_{key}"]
            # Check with 'test_' prefix (used in final evaluation)
            elif f"test_{key}" in all_results[task]:
                score = all_results[task][f"test_{key}"]

            if score is not None:
                scores.append(score)
                print(f"  {task:<10} {key:<26} {score:>8.4f}")
            else:
                # If we can't find the specific metric, try to find any metric
                available_metrics = [
                    k
                    for k in all_results[task].keys()
                    if isinstance(all_results[task][k], (int, float))
                ]
                if available_metrics:
                    # Try to find a metric that matches the expected pattern
                    matching_metrics = [k for k in available_metrics if key in k]
                    if matching_metrics:
                        score = all_results[task][matching_metrics[0]]
                        scores.append(score)
                        print(f"  {task:<10} {key:<26} {score:>8.4f}")
                    else:
                        # Use the first available metric as fallback
                        score = all_results[task][available_metrics[0]]
                        scores.append(score)
                        print(f"  {task:<10} {available_metrics[0]:<26} {score:>8.4f}")
                else:
                    print(f"  {task:<10} {'ERROR / SKIPPED':<26}")
        else:
            print(f"  {task:<10} {'ERROR / SKIPPED':<26}")

    if scores:
        print(f"  {'-' * 44}")
        print(f"  {'GLUE Avg':<10} {'':<26} {sum(scores) / len(scores):>8.4f}")
    print("=" * 70)
    print("=" * 70)


def build_optimizer_and_scheduler(
    model,
    adapter_lr: float,
    head_lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
    adapter_type: str = "lora",
    scheduler_type: str = "linear",
) -> tuple:
    """
    Split trainable parameters into two groups:
      - 'head'    : classifier.* — trained at head_lr
      - 'adapter' : everything else — trained at adapter_lr

    Weight decay is applied only to non-bias, non-norm parameters in both groups.
    For bitfit, classifier biases are separated from model biases to allow
    independent learning rates (same as other adapters).
    """
    if adapter_type == "head":
        head_params = [
            p
            for n, p in model.named_parameters()
            if "classifier" in n and p.requires_grad
        ]
        if not head_params:
            return None, None

        head_wd = [p for p in head_params if p.ndim != 1]
        head_no_wd = [p for p in head_params if p.ndim == 1]

        param_groups = [
            {"params": head_wd, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_wd, "lr": head_lr, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = AdamW(param_groups)

        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:  # default to linear for retro-compatibility
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        return optimizer, scheduler

    head_params, adapter_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            adapter_params.append(param)

    def split_wd(params):
        wd, no_wd = [], []
        for p in params:
            (no_wd if p.ndim == 1 else wd).append(p)
        return wd, no_wd

    head_wd, head_no_wd = split_wd(head_params)

    # Remove split_wd for GPart, apply WD unconditionally to theta_d
    if adapter_type == "gpart":
        param_groups = [
            {"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay},
            {"params": head_wd, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_wd, "lr": head_lr, "weight_decay": 0.0},
        ]
    else:
        adapter_wd, adapter_no_wd = split_wd(adapter_params)
        param_groups = [
            {"params": adapter_wd, "lr": adapter_lr, "weight_decay": weight_decay},
            {"params": adapter_no_wd, "lr": adapter_lr, "weight_decay": 0.0},
            {"params": head_wd, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_wd, "lr": head_lr, "weight_decay": 0.0},
        ]

    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(param_groups)

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:  # default to linear for retro-compatibility
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    return optimizer, scheduler
