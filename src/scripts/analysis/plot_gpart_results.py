#!/usr/bin/env python3
"""
Plot GPart RoBERTa Base SST2 results using seaborn.

This script creates a line plot showing accuracy vs # parameters (d)
for GPart fine-tuning experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(filepath):
    """Load and parse the CSV data."""
    df = pd.read_csv(filepath)
    return df


def create_plot(df, output_path=None):
    """Create the seaborn plot."""
    # Set up the plotting style
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    # Create the plot with log scale for x-axis
    plt.semilogx(df["d"], df["accuracy (sst2)"], "o-", linewidth=2, markersize=8)

    # Customize the plot
    plt.xlabel("# Parameters", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.title(
        "SST-2 Accuracy vs # Parameters",
        fontsize=20,
        # fontweight="bold",
    )

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Set x-axis to show all d values as ticks
    plt.xticks(df["d"], rotation=45, ha="right")

    # Format x-axis labels to show values nicely
    ax = plt.gca()
    ax.set_xticklabels([f"{int(x):,}" for x in df["d"]])

    # Add some padding to y-axis
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.5, y_max + 0.5)

    # Add value labels on points for better readability
    for i, (d, acc) in enumerate(zip(df["d"], df["accuracy (sst2)"])):
        plt.annotate(
            f"{acc:.1f}",
            (d, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function to load data and create plot."""
    # File paths
    input_file = "results/gpart_roberta_large_sst2.csv"
    output_file = "figures/gpart_roberta_large_sst2_plot.png"

    try:
        # Load the data
        print(f"Loading data from {input_file}...")
        df = load_data(input_file)
        print(f"Data loaded successfully:")
        print(df)
        print()

        # Create the plot
        print(f"Creating plot...")
        create_plot(df, output_file)
        print("Plot created successfully!")

    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        print("Please make sure the CSV file exists in the results directory.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
