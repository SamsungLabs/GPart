"""
Gradient Landscape Visualization for GPart vs Uni-LoRA

Produces side-by-side 2D contour plots of the loss landscape around the
converged solution for GPart and Uni-LoRA, using the filter normalization
method (Li et al., 2018) adapted to the trainable subspace.

Usage:
    python src/scripts/loss_landscape.py \
        --task sst2 \
        --model_size base \
        --gpart_checkpoint experiments/logs/roberta_glue_gpart/sst2/seed_0/adapter/best_model \
        --unilora_checkpoint experiments/logs/roberta_glue_unilora/sst2/seed_0/adapter/best_model \
        --d 23000 \
        --grid_size 30 \
        --alpha_range 0.5 \
        --n_seeds 3 \
        --output_dir figures/loss_landscape \
        --seed 0
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

# ── PEFT ──────────────────────────────────────────────────────────────────────
from peft import PeftModel

# ── Utils ─────────────────────────────────────────────────────────────────────
from configs.task_configs import GLUE_TASK_METADATA, REGRESSION_TASKS
from utils.data_loader_utils import create_tokenize_fn


def load_val_loader(
    task_name: str,
    tokenizer,
    batch_size: int = 128,
    max_length: int = 128,
    max_samples: int = 1024,
    seed: int = 0,
):
    """Load a fixed validation batch for loss evaluation."""
    raw = load_dataset("glue", task_name)
    tok_fn = create_tokenize_fn(tokenizer, GLUE_TASK_METADATA[task_name].text_keys, max_length)
    remove_cols = [
        c
        for c in raw["validation"].column_names
        if c not in ("label", "input_ids", "attention_mask", "token_type_ids")
    ]
    ds = raw["validation"].map(tok_fn, batched=True, remove_columns=remove_cols)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    # Subsample for speed
    if len(ds) > max_samples:
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        ds = ds.select(indices[:max_samples])

    collator = DataCollatorWithPadding(tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)


# =============================================================================
# Trainable-parameter extraction
# =============================================================================


def set_trainable_vector(model: torch.nn.Module, vec: torch.Tensor):
    """Write a flat vector back into the model's trainable parameters."""
    device = next(iter(p for p in model.parameters() if p.requires_grad)).device
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            p.data.copy_(vec[offset : offset + n].view(p.shape).to(device))
            offset += n


# =============================================================================
# Loss evaluation
# =============================================================================


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module, loader: DataLoader, is_regression: bool, device: str
) -> float:
    model.eval()
    total, steps = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        steps += 1
    return total / steps if steps > 0 else float("nan")


# =============================================================================
# Loss surface computation
# =============================================================================


def compute_surface(
    model: torch.nn.Module,
    theta_star: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    alphas: np.ndarray,
    betas: np.ndarray,
    loader: DataLoader,
    is_regression: bool,
    device: str,
) -> np.ndarray:
    """Evaluate loss on a 2-D grid of perturbations around theta_star."""
    surface = np.zeros((len(alphas), len(betas)), dtype=np.float32)
    for i, a in tqdm(enumerate(alphas), desc="Computing surface", total=len(alphas)):
        for j, b in enumerate(betas):
            theta_perturbed = theta_star + a * d1 + b * d2
            set_trainable_vector(model, theta_perturbed)
            surface[i, j] = eval_loss(model, loader, is_regression, device)
    # Restore original weights
    set_trainable_vector(model, theta_star)
    return surface


def sample_direction(theta_star: torch.Tensor, seed: int) -> torch.Tensor:
    """Sample a filter-normalised random direction with a given seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    delta = torch.randn(theta_star.shape, generator=gen)
    # Scale direction to have the same norm as theta_star
    delta = delta / delta.norm() * theta_star.norm()
    return delta


def average_surfaces(
    model, theta_star, alphas, betas, loader, is_regression, device, n_seeds, base_seed
):
    """Average loss surface over multiple random direction seeds."""
    surfaces = []
    for s in range(n_seeds):
        print(f"Computing average loss surface for seed {s}/{n_seeds-1}")
        d1 = sample_direction(theta_star, seed=base_seed + s * 1000)
        d2 = sample_direction(theta_star, seed=base_seed + s * 1000 + 1)
        surf = compute_surface(
            model, theta_star, d1, d2, alphas, betas, loader, is_regression, device
        )
        surfaces.append(surf)
    return np.mean(surfaces, axis=0)


# =============================================================================
# Plotting
# =============================================================================


def plot_landscape(
    surfaces: dict,
    alphas: np.ndarray,
    betas: np.ndarray,
    task_name: str,
    output_dir: str,
):
    """
    Plot side-by-side contour plots for each method.
    All subplots share the same colour scale for a fair visual comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    methods = list(surfaces.keys())
    n = len(methods)

    # Shared colour range
    vmin = min(s.min() for s in surfaces.values())
    vmax = max(s.max() for s in surfaces.values())
    levels = np.linspace(vmin, vmax, 25)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        surf = surfaces[method]
        cf = ax.contourf(betas, alphas, surf, levels=levels, cmap="coolwarm")
        ax.contour(
            betas, alphas, surf, levels=levels, colors="k", linewidths=0.3, alpha=0.4
        )
        ax.set_title(method, fontsize=14, fontweight="bold")
        ax.set_xlabel(r"$\beta$  (direction $\delta_2$)", fontsize=11)
        ax.set_ylabel(r"$\alpha$  (direction $\delta_1$)", fontsize=11)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_aspect("equal")

    fig.colorbar(cf, ax=axes, label="Validation loss", shrink=0.8)
    fig.suptitle(
        f"Loss landscape — {task_name.upper()}\n"
        f"(filter-normalised random directions, averaged over {args_global.n_seeds} seeds)",
        fontsize=12,
    )

    out_png = os.path.join(output_dir, f"loss_landscape_{task_name}.png")
    out_pdf = os.path.join(output_dir, f"loss_landscape_{task_name}.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  →  {out_png}")
    print(f"Saved  →  {out_pdf}")


# =============================================================================
# Main
# =============================================================================

args_global = None  # set in main for access in plot_landscape


def load_model(checkpoint_path: str, base_model_name: str, task_name: str):
    meta = GLUE_TASK_METADATA[task_name]
    is_regression = task_name in REGRESSION_TASKS
    problem_type = "regression" if is_regression else "single_label_classification"
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=meta.num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=True)
    # Ensure adapter parameters are marked trainable
    # (PeftModel.from_pretrained with is_trainable=False freezes them by default)
    for name, param in model.named_parameters():
        if (
            "lora_" in name
            or "vera_" in name
            or "theta" in name
            or "gpart" in name.lower()
        ):
            param.requires_grad_(True)

    # Freeze classifier head — landscape should reflect adapter subspace only
    for param in model.classifier.parameters():
        param.requires_grad_(False)

    return model


def get_trainable_vector(model: torch.nn.Module) -> torch.Tensor:
    """Flatten trainable adapter parameters into a 1-D tensor (CPU).
    Falls back to ALL parameters if no requires_grad params are found
    (e.g. merged adapters), which should not happen after load_model above.
    """
    params = [p.detach().cpu().flatten() for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "No trainable parameters found. The adapter may have been merged "
            "into the base model. Reload with is_trainable=True."
        )
    return torch.cat(params)


def parse_args():
    p = argparse.ArgumentParser(
        description="Loss landscape visualisation: GPart vs Uni-LoRA"
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    p.add_argument(
        "--gpart_checkpoint",
        required=True,
        help="Path to saved GPart PEFT checkpoint (best_model dir).",
    )
    p.add_argument(
        "--unilora_checkpoint",
        required=True,
        help="Path to saved Uni-LoRA PEFT checkpoint (best_model dir).",
    )
    p.add_argument(
        "--extra_checkpoints",
        nargs="*",
        default=[],
        help="Optional additional PEFT checkpoint paths to compare.",
    )
    p.add_argument(
        "--extra_labels",
        nargs="*",
        default=[],
        help="Labels for extra checkpoints (same order).",
    )

    # ── Task / model ─────────────────────────────────────────────────────────
    p.add_argument(
        "--task",
        default="sst2",
        choices=list(GLUE_TASK_METADATA.keys()),
        help="GLUE task for evaluation.",
    )
    p.add_argument("--model_size", choices=["base", "large"], default="base")
    p.add_argument(
        "--base_model",
        default=None,
        help="Override base model name (default: roberta-{model_size}).",
    )

    # ── Grid settings ────────────────────────────────────────────────────────
    p.add_argument(
        "--grid_size",
        type=int,
        default=30,
        help="Number of grid points per axis (grid_size x grid_size).",
    )
    p.add_argument(
        "--alpha_range",
        type=float,
        default=0.5,
        help="Half-range for alpha and beta axes.",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of random direction seeds to average over.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=1024,
        help="Max validation samples to use for loss evaluation.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=128)

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="figures/loss_landscape")
    return p.parse_args()


def main():
    global args_global
    args = parse_args()
    args_global = args

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_model_name = args.base_model or f"roberta-{args.model_size}"
    is_regression = args.task in REGRESSION_TASKS
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # ── Validation loader ────────────────────────────────────────────────────
    print(f"Loading validation data for {args.task.upper()} ...")
    loader = load_val_loader(
        args.task,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # ── Grid ────────────────────────────────────────────────────────────────
    alphas = np.linspace(-args.alpha_range, args.alpha_range, args.grid_size)
    betas = np.linspace(-args.alpha_range, args.alpha_range, args.grid_size)

    # ── Checkpoints to evaluate ──────────────────────────────────────────────
    checkpoints = {
        "GPart": args.gpart_checkpoint,
        "Uni-LoRA": args.unilora_checkpoint,
    }
    for label, ckpt in zip(args.extra_labels, args.extra_checkpoints):
        checkpoints[label] = ckpt

    # ── Compute surfaces ─────────────────────────────────────────────────────
    surfaces = {}
    for label, ckpt_path in checkpoints.items():
        print(f"\n[{label}] Loading checkpoint: {ckpt_path}")
        model = load_model(ckpt_path, base_model_name, args.task)
        model.to(device)

        theta_star = get_trainable_vector(model)
        d = theta_star.numel()
        print(f"[{label}] Trainable parameters (d): {d:,}")

        # Baseline loss at theta*
        baseline = eval_loss(model, loader, is_regression, device)
        print(f"[{label}] Baseline validation loss: {baseline:.4f}")

        print(
            f"[{label}] Computing loss surface "
            f"({args.grid_size}x{args.grid_size} grid, {args.n_seeds} direction seeds) ..."
        )
        surf = average_surfaces(
            model,
            theta_star,
            alphas,
            betas,
            loader,
            is_regression,
            device,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
        )
        surfaces[label] = surf
        print(f"[{label}] Loss range: [{surf.min():.4f}, {surf.max():.4f}]")

        # Save raw surface for later re-plotting without re-running
        np.save(
            os.path.join(
                args.output_dir,
                f"surface_{args.task}_{label.lower().replace('-','_').replace(' ','_')}.npy",
            ),
            surf,
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_landscape(surfaces, alphas, betas, args.task, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
