"""
Generate figures for cross-model validation analysis.

Produces:
  Figure 1: Failure Mode Space (WAcc × Fluency scatter)
  Figure 2: Relative Change from Baseline (bar chart)
  Figure 3: Repetition Rates (bar chart)
  Figure 4: Dose-Response (optional, requires sweep data)

Usage:
    python make_plots.py \
        --per_utterance_csv /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/per_utterance_metrics_whisper.csv \
        --aggregate_csv /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/aggregate_metrics_whisper.csv \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Style constants
CONDITION_COLORS = {
    "base": "#2ecc71",
    "UU": "#e74c3c",
    "UR": "#9b59b6",
    "RR": "#3498db",
    "RU": "#f39c12",
}
CONDITION_MARKERS = {
    "base": "o",
    "UU": "s",
    "UR": "D",
    "RR": "^",
    "RU": "v",
}

BAR_COLORS = {
    "UU": "#e74c3c",
    "UR": "#9b59b6",
    "RR": "#3498db",
    "RU": "#f39c12",
}


def figure1_failure_mode_space(df, output_dir, primary_norm_col=None):
    """
    Scatter plot: x-axis = WAcc, y-axis = normalized sentence score.
    Points colored by noise condition.
    """
    print("Generating Figure 1: Failure Mode Space...", flush=True)

    # Find the primary fluency column
    if primary_norm_col is None:
        norm_cols = [c for c in df.columns if c.startswith("normalized_sentence_score_")]
        if not norm_cols:
            print("  No normalized_sentence_score column found, skipping", flush=True)
            return
        # Prefer strong LM
        strong_cols = [c for c in norm_cols if "gpt2" not in c.lower()]
        primary_norm_col = strong_cols[0] if strong_cols else norm_cols[0]
    lm_name = primary_norm_col.replace("normalized_sentence_score_", "")

    fig, ax = plt.subplots(figsize=(10, 7))

    for cond in sorted(df["noise_condition"].unique()):
        cond_df = df[df["noise_condition"] == cond]
        color = CONDITION_COLORS.get(cond, "#95a5a6")
        marker = CONDITION_MARKERS.get(cond, "o")
        alpha = 0.4 if cond == "base" else 0.5

        ax.scatter(
            cond_df["wacc"],
            cond_df[primary_norm_col],
            c=color, marker=marker, alpha=alpha,
            label=f"{cond} (n={len(cond_df)})",
            s=20, edgecolors="none",
        )

    ax.set_xlabel("WAcc (1 - WER)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Normalized Sentence Score ({lm_name})", fontsize=13, fontweight="bold")
    ax.set_title("Failure Mode Space: WAcc × Fluency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Draw threshold lines if base exists
    base_df = df[df["noise_condition"] == "base"]
    if len(base_df) > 0:
        wacc_thresh = base_df["wacc"].mean()
        flu_thresh = base_df[primary_norm_col].mean()
        ax.axvline(x=wacc_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=flu_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate("hallucination-like\nregion", xy=(0.02, 0.98), fontsize=9,
                     color="gray", ha="left", va="top")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "failure_mode_space.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "failure_mode_space.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved failure_mode_space.png/pdf", flush=True)


def figure2_relative_change(agg_df, output_dir):
    """
    Bar plot showing delta WAcc, delta hallucination rate, delta repetition rate.
    """
    print("Generating Figure 2: Relative Change from Baseline...", flush=True)

    # Check for necessary columns
    base_row = agg_df[agg_df["condition"] == "base"]
    if len(base_row) == 0:
        print("  No baseline row, skipping Figure 2", flush=True)
        return

    noise_conds = [c for c in ["UU", "UR", "RR", "RU"] if c in agg_df["condition"].values]
    if not noise_conds:
        print("  No noise conditions found, skipping Figure 2", flush=True)
        return

    # Compute deltas
    base = base_row.iloc[0]
    metrics_for_delta = []
    metric_labels = []

    if "mean_wacc" in agg_df.columns:
        base_wacc = base["mean_wacc"]
        deltas_wacc = []
        for cond in noise_conds:
            row = agg_df[agg_df["condition"] == cond].iloc[0]
            deltas_wacc.append(row["mean_wacc"] - base_wacc)
        metrics_for_delta.append(("Δ WAcc", deltas_wacc, "mean_wacc"))

    if "hallucination_like_rate" in agg_df.columns:
        base_hall = base["hallucination_like_rate"]
        deltas_hall = []
        for cond in noise_conds:
            row = agg_df[agg_df["condition"] == cond].iloc[0]
            deltas_hall.append(row["hallucination_like_rate"] - base_hall)
        metrics_for_delta.append(("Δ Hall. Rate", deltas_hall, "hallucination_like_rate"))

    if "trigram_rep_rate" in agg_df.columns:
        base_tri = base["trigram_rep_rate"]
        deltas_tri = []
        for cond in noise_conds:
            row = agg_df[agg_df["condition"] == cond].iloc[0]
            deltas_tri.append(row["trigram_rep_rate"] - base_tri)
        metrics_for_delta.append(("Δ Trigram Rep.", deltas_tri, "trigram_rep_rate"))

    if not metrics_for_delta:
        print("  No delta metrics available", flush=True)
        return

    n_metrics = len(metrics_for_delta)
    n_conds = len(noise_conds)
    x = np.arange(n_metrics)
    bar_width = 0.2
    offsets = np.linspace(-bar_width * (n_conds - 1) / 2, bar_width * (n_conds - 1) / 2, n_conds)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cond in enumerate(noise_conds):
        values = [metric_data[1][i] for metric_data in metrics_for_delta]
        color = BAR_COLORS.get(cond, "#95a5a6")
        ax.bar(x + offsets[i], values, bar_width, label=cond, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Δ from Baseline", fontsize=13, fontweight="bold")
    ax.set_title("Relative Change from Whisper Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics_for_delta], fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "relative_change_from_baseline.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "relative_change_from_baseline.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved relative_change_from_baseline.png/pdf", flush=True)


def figure3_repetition_rates(agg_df, output_dir):
    """
    Bar plot showing trigram + four-gram repetition rate per condition.
    """
    print("Generating Figure 3: Repetition Rates...", flush=True)

    conditions = [c for c in ["base", "UU", "UR", "RR", "RU"] if c in agg_df["condition"].values]
    if not conditions:
        print("  No conditions found, skipping Figure 3", flush=True)
        return

    tri_rates = []
    four_rates = []
    for cond in conditions:
        row = agg_df[agg_df["condition"] == cond].iloc[0]
        tri_rates.append(row.get("trigram_rep_rate", 0))
        four_rates.append(row.get("fourgram_rep_rate", 0))

    x = np.arange(len(conditions))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - bar_width / 2, tri_rates, bar_width,
                   label="Trigram Repetition Rate", color="#e74c3c", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, four_rates, bar_width,
                   label="Four-gram Repetition Rate", color="#2c3e50", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Noise Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Repetition Rate (proportion of utterances)", fontsize=13, fontweight="bold")
    ax.set_title("N-gram Repetition by Noise Condition", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(tri_rates), max(four_rates)) * 1.3 + 0.01)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "repetition_rates.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "repetition_rates.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved repetition_rates.png/pdf", flush=True)


def figure4_dose_response(df, output_dir, primary_norm_col=None):
    """
    Optional: dose-response curves for sweep experiments.
    Only generated if noise_ratio varies (i.e., sweep data is present).
    """
    noise_ratios = df["noise_ratio"].unique()
    if len(noise_ratios) <= 1:
        print("Skipping Figure 4: No dose-response data (only one noise ratio in data)", flush=True)
        return

    print("Generating Figure 4: Dose-Response...", flush=True)

    if primary_norm_col is None:
        norm_cols = [c for c in df.columns if c.startswith("normalized_sentence_score_")]
        if not norm_cols:
            return
        strong_cols = [c for c in norm_cols if "gpt2" not in c.lower()]
        primary_norm_col = strong_cols[0] if strong_cols else norm_cols[0]

    noise_types = ["UU", "UR", "RR", "RU"]
    noise_colors = {"UU": "#e74c3c", "UR": "#9b59b6", "RR": "#3498db", "RU": "#f39c12"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("wacc", "WAcc", "WAcc vs Noise Ratio", False),
        ("hallucination_like", "Hall. Rate", "Hallucination Rate vs Noise Ratio", False),
        ("trigram_rep_count", "Trigram Rep. Count", "Trigram Repetition vs Noise Ratio", False),
    ]

    for ax, (col, ylabel, title, is_rep) in zip(axes, metrics):
        for ntype in noise_types:
            type_df = df[df["noise_condition"] == ntype]
            if len(type_df) == 0:
                continue

            # Aggregate by noise_ratio
            ratios = sorted(type_df["noise_ratio"].unique())
            means = []
            for r in ratios:
                r_df = type_df[type_df["noise_ratio"] == r]
                if col == "hallucination_like":
                    means.append(r_df.get("hallucination_like", pd.Series([0])).mean())
                else:
                    means.append(r_df[col].mean() if col in r_df.columns else 0)

            color = noise_colors.get(ntype, "#95a5a6")
            ax.plot(ratios, means, color=color, marker="o", linewidth=2, markersize=7, label=ntype)

        ax.set_xlabel("Noise Ratio", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "dose_response.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "dose_response.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved dose_response.png/pdf", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate validation plots")
    parser.add_argument("--per_utterance_csv", type=str, required=True,
                        help="Path to per_utterance_metrics_whisper.csv")
    parser.add_argument("--aggregate_csv", type=str, required=True,
                        help="Path to aggregate_metrics_whisper.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...", flush=True)
    df = pd.read_csv(args.per_utterance_csv)
    agg_df = pd.read_csv(args.aggregate_csv)
    print(f"  Per-utterance: {len(df)} rows", flush=True)
    print(f"  Aggregate: {len(agg_df)} rows", flush=True)

    # Detect primary LM column
    norm_cols = [c for c in df.columns if c.startswith("normalized_sentence_score_")]
    if norm_cols:
        strong_cols = [c for c in norm_cols if "gpt2" not in c.lower() and "small" not in c.lower()]
        primary_norm_col = strong_cols[0] if strong_cols else norm_cols[0]
    else:
        primary_norm_col = None

    # Generate figures
    figure1_failure_mode_space(df, args.output_dir, primary_norm_col)
    figure2_relative_change(agg_df, args.output_dir)
    figure3_repetition_rates(agg_df, args.output_dir)
    figure4_dose_response(df, args.output_dir, primary_norm_col)

    print(f"\nAll plots saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
