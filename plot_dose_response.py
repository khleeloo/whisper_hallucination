"""
Plot dose-response curves showing how noise ratio affects ASR failure modes.

Reads evaluation results from the sweep experiment and plots:
1. WER vs noise ratio (per noise type)
2. Repetition rate vs noise ratio
3. Plausibility score vs noise ratio

Usage:
    python plot_dose_response.py \
        --results_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination \
        --output_dir /home/rmfrieske/whisper_hallucination/plots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


NOISE_TYPES = ["uu", "rr"]
NOISE_RATIOS = [1, 2, 5, 10, 20, 50]
TYPE_LABELS = {"uu": "Unique-Unique (UU)", "rr": "Repeat-Repeat (RR)"}
TYPE_COLORS = {"uu": "#e74c3c", "rr": "#3498db"}
TYPE_MARKERS = {"uu": "o", "rr": "s"}


def load_results(results_dir):
    """Load evaluation JSON results for base + sweep configs."""
    data = {}

    # Base (0% noise)
    base_path = os.path.join(results_dir, "base", "eval_results.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            data[("base", 0)] = json.load(f)

    # Sweep configs
    for ntype in NOISE_TYPES:
        for ratio in NOISE_RATIOS:
            config = f"sweep_{ntype}_{ratio:02d}"
            path = os.path.join(results_dir, config, "eval_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[(ntype, ratio)] = json.load(f)

    # Also check the original 8% experiments
    for ntype in NOISE_TYPES:
        orig_path = os.path.join(results_dir, ntype, "eval_results.json")
        if os.path.exists(orig_path) and (ntype, 8) not in data:
            with open(orig_path) as f:
                data[(ntype, 8)] = json.load(f)

    return data


def plot_metric(ax, data, metric_key, ylabel, title):
    """Plot a single metric vs noise ratio for all noise types."""
    for ntype in NOISE_TYPES:
        ratios = []
        values = []

        # Add base (0%)
        if ("base", 0) in data and metric_key in data[("base", 0)]:
            ratios.append(0)
            values.append(data[("base", 0)][metric_key])

        # Add sweep points
        for ratio in sorted(NOISE_RATIOS + [8]):  # include 8% if available
            if (ntype, ratio) in data and metric_key in data[(ntype, ratio)]:
                ratios.append(ratio)
                values.append(data[(ntype, ratio)][metric_key])

        if ratios:
            ax.plot(ratios, values,
                    color=TYPE_COLORS[ntype],
                    marker=TYPE_MARKERS[ntype],
                    linewidth=2, markersize=8,
                    label=TYPE_LABELS[ntype])

    ax.set_xlabel("Noise Ratio (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 52)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.results_dir)

    if not data:
        print("No results found. Run evaluation first.")
        return

    print(f"Loaded results for {len(data)} configs:")
    for key in sorted(data.keys()):
        print(f"  {key}: {list(data[key].keys())}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("wer", "WER (%)", "Word Error Rate vs Noise Ratio"),
        ("repetition_rate", "Repetition Rate", "N-gram Repetition vs Noise Ratio"),
        ("avg_plausibility", "Plausibility Score", "Output Plausibility vs Noise Ratio"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        plot_metric(ax, data, key, ylabel, title)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "dose_response_curves.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # Also save PNG
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
