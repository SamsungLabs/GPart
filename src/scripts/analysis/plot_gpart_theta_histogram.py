"""
GPart theta_d Histogram Visualiser

Loads a trained GPart checkpoint and plots the distribution of the d-dimensional
trainable vector theta_d, highlighting outlier components.

Usage:
    python src/scripts/gpart_theta_histogram.py \
        --checkpoint experiments/logs/roberta_glue_gpart/sst2/seed_0/adapter/best_model \
        --task sst2 \
        --model_size base \
        --output_dir figures/theta_histogram \
        --outlier_threshold 3.0
"""

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, set_seed

from peft import PeftModel
from configs.task_configs import GLUE_TASK_METADATA, REGRESSION_TASKS

# =============================================================================
# Helpers
# =============================================================================


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
    # Freeze classifier head — we only want adapter parameters
    for param in model.classifier.parameters():
        param.requires_grad_(False)
    return model


def get_trainable_vector(model: torch.nn.Module) -> torch.Tensor:
    params = [p.detach().cpu().flatten() for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "No trainable parameters found. The adapter may have been merged. "
            "Reload with is_trainable=True."
        )
    return torch.cat(params)


# =============================================================================
# Outlier analysis
# =============================================================================


def find_outliers(theta: np.ndarray, threshold: float):
    """
    Flag components whose absolute value exceeds `threshold` standard
    deviations from the mean.
    Returns indices, values, and z-scores of outliers.
    """
    mean = theta.mean()
    std = theta.std()
    z = (theta - mean) / (std + 1e-12)
    mask = np.abs(z) > threshold
    indices = np.where(mask)[0]
    return indices, theta[indices], z[indices]


# =============================================================================
# Plotting
# =============================================================================


def plot_histogram(
    theta: np.ndarray,
    outlier_indices: np.ndarray,
    outlier_values: np.ndarray,
    outlier_zscores: np.ndarray,
    task_name: str,
    model_size: str,
    threshold: float,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    mean, std = theta.mean(), theta.std()
    d = len(theta)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Main histogram ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    n_bins = min(200, max(50, d // 100))
    counts, bin_edges, patches = ax1.hist(
        theta, bins=n_bins, color="#3a7ebf", edgecolor="none", alpha=0.85
    )

    # Colour outlier bins red
    for patch, left in zip(patches, bin_edges[:-1]):
        right = left + (bin_edges[1] - bin_edges[0])
        if left < mean - threshold * std or right > mean + threshold * std:
            patch.set_facecolor("#c0392b")
            patch.set_alpha(0.9)

    ax1.axvline(
        mean, color="black", linestyle="--", linewidth=1.2, label=f"mean = {mean:.4f}"
    )
    ax1.axvline(
        mean + threshold * std,
        color="#c0392b",
        linestyle=":",
        linewidth=1.0,
        label=f"±{threshold}σ threshold",
    )
    ax1.axvline(mean - threshold * std, color="#c0392b", linestyle=":", linewidth=1.0)
    ax1.set_xlabel(r"$\theta_d$ component value", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(
        rf"Distribution of $\theta_d$ components — {task_name.upper()} "
        rf"(RoBERTa-{model_size},  $d={d:,}$)",
        fontsize=13,
    )
    ax1.legend(fontsize=10)
    stats_text = (
        f"mean={mean:.4f}  std={std:.4f}  "
        f"min={theta.min():.4f}  max={theta.max():.4f}\n"
        f"outliers (|z|>{threshold}): {len(outlier_indices)} / {d} "
        f"({100*len(outlier_indices)/d:.2f}%)"
    )
    ax1.text(
        0.02,
        0.97,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # ── 2. Absolute-value CDF ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sorted_abs = np.sort(np.abs(theta))
    cdf = np.arange(1, d + 1) / d
    ax2.plot(sorted_abs, cdf, color="#3a7ebf", linewidth=1.2)
    ax2.axvline(
        threshold * std,
        color="#c0392b",
        linestyle=":",
        linewidth=1.0,
        label=f"{threshold}σ",
    )
    ax2.set_xlabel(r"|$\theta_d$| value", fontsize=11)
    ax2.set_ylabel("CDF", fontsize=11)
    ax2.set_title("Cumulative distribution (|values|)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(left=0)

    # ── 3. Outlier scatter ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if len(outlier_indices) > 0:
        sc = ax3.scatter(
            outlier_indices,
            outlier_values,
            c=outlier_zscores,
            cmap="RdBu_r",
            s=12,
            alpha=0.8,
            vmin=-np.abs(outlier_zscores).max(),
            vmax=np.abs(outlier_zscores).max(),
        )
        plt.colorbar(sc, ax=ax3, label="z-score")
        ax3.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    else:
        ax3.text(
            0.5,
            0.5,
            "No outliers found",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
    ax3.set_xlabel("Component index", fontsize=11)
    ax3.set_ylabel(r"$\theta_d$ value", fontsize=11)
    ax3.set_title(
        f"Outlier components (|z| > {threshold}): {len(outlier_indices)}", fontsize=11
    )

    out_png = os.path.join(output_dir, f"theta_histogram_{task_name}.png")
    # out_pdf = os.path.join(output_dir, f"theta_histogram_{task_name}.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  →  {out_png}")
    # print(f"Saved  →  {out_pdf}")


# =============================================================================
# Outlier report
# =============================================================================


def print_outlier_report(
    theta: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    zscores: np.ndarray,
    threshold: float,
):
    mean, std = theta.mean(), theta.std()
    print("\n" + "=" * 60)
    print(f"THETA_D SUMMARY")
    print("=" * 60)
    print(f"  d (dimension)  : {len(theta):>10,}")
    print(f"  mean           : {mean:>+10.6f}")
    print(f"  std            : {std:>10.6f}")
    print(f"  min            : {theta.min():>+10.6f}")
    print(f"  max            : {theta.max():>+10.6f}")
    print(f"  median         : {np.median(theta):>+10.6f}")
    print(
        f"  kurtosis (excess): {float(((theta - mean)**4).mean() / std**4 - 3):>+8.4f}"
    )
    print(f"  skewness       : {float(((theta - mean)**3).mean() / std**3):>+8.4f}")
    print()
    print(
        f"OUTLIERS  (|z| > {threshold}): {len(indices)} / {len(theta)} "
        f"({100*len(indices)/len(theta):.3f}%)"
    )
    print("-" * 60)

    if len(indices) == 0:
        print("  No outliers detected.")
    else:
        # Sort by descending |z-score|
        order = np.argsort(-np.abs(zscores))
        n_show = min(20, len(indices))
        print(f"  Top-{n_show} outliers by |z-score|:")
        print(f"  {'Index':>10}  {'Value':>12}  {'z-score':>10}  {'Sign':>6}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*6}")
        for i in order[:n_show]:
            sign = "pos" if values[i] > 0 else "neg"
            print(
                f"  {indices[i]:>10,}  {values[i]:>+12.6f}  {zscores[i]:>+10.4f}  {sign:>6}"
            )

        # Positive / negative split
        pos = (values > 0).sum()
        neg = (values < 0).sum()
        print()
        print(f"  Positive outliers: {pos}  |  Negative outliers: {neg}")

    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Histogram and outlier analysis of GPart theta_d vector."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to saved GPart PEFT checkpoint (best_model dir).",
    )
    p.add_argument(
        "--task",
        default="sst2",
        choices=["cola", "sst2", "mrpc", "stsb", "qnli", "rte"],
    )
    p.add_argument("--model_size", choices=["base", "large"], default="base")
    p.add_argument(
        "--base_model",
        default=None,
        help="Override base model name (default: roberta-{model_size}).",
    )
    p.add_argument(
        "--outlier_threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for flagging outlier components.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="figures/theta_histogram")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    base_model_name = args.base_model or f"roberta-{args.model_size}"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, base_model_name, args.task)
    print(model.peft_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    theta = get_trainable_vector(model).numpy()
    print(f"theta_d dimension: {len(theta):,}")

    indices, values, zscores = find_outliers(theta, args.outlier_threshold)

    print_outlier_report(theta, indices, values, zscores, args.outlier_threshold)

    plot_histogram(
        theta,
        indices,
        values,
        zscores,
        task_name=args.task,
        model_size=args.model_size,
        threshold=args.outlier_threshold,
        output_dir=args.output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
