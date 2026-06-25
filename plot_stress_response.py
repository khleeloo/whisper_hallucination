"""
Generate stress-response plots for Whisper hallucination experiments.

Figures produced:
  2. Base-normalized response matrix heatmaps.
  3. Perturbation dose-response curves.
  4. Wrong-but-fluent per-utterance scatter plots.
  5. Catastrophic error tail plots.

The script reads summary JSON files named like:
  results_rr_64pct_full_noise_amp0.5_dur0.0.json

and optional per-utterance TSV files named like:
  details_rr_64pct_full_noise_amp0.5_dur0.0.tsv

Usage:
    python plot_stress_response.py \
        --stress_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_64pct \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_plots
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG_COLORS = {
    "base": "#2ca25f",
    "rr": "#3182bd",
    "ru": "#f39c12",
    "uu": "#de2d26",
    "ur": "#756bb1",
}
CONFIG_MARKERS = {
    "base": "o",
    "rr": "s",
    "ru": "^",
    "uu": "D",
    "ur": "v",
}

SUMMARY_RE = re.compile(r"^results_(?P<config>.+)_(?P<perturbation>(?:none|.+_amp[^_]+_dur[^_]+))\.json$")
PERTURB_RE = re.compile(r"^(?P<family>.+)_amp(?P<amplitude>[^_]+)_dur(?P<duration>[^_]+)$")


def config_family(config):
    lowered = str(config).lower()
    if lowered == "base" or lowered.startswith("base"):
        return "base"
    return lowered.split("_")[0]


def noise_pct(config):
    if config_family(config) == "base":
        return 0
    match = re.search(r"(\d+)pct", str(config))
    return int(match.group(1)) if match else np.nan


def parse_perturbation(perturbation):
    if perturbation == "none":
        return "none", 0.0, 0.0, "none"

    match = PERTURB_RE.match(perturbation)
    if not match:
        return perturbation, np.nan, np.nan, perturbation


    family = match.group("family")
    amplitude = float(match.group("amplitude"))
    duration = float(match.group("duration"))
    label = f"{family}\namp={amplitude:g}, dur={duration:g}"
    return family, amplitude, duration, label


def sort_key_for_perturbation(row):
    family_order = {"none": 0, "onset_noise": 1, "full_noise": 2, "reverb": 3}
    return (
        family_order.get(row["perturb_family"], 99),
        row["perturb_amplitude"] if pd.notna(row["perturb_amplitude"]) else -1,
        row["perturb_duration"] if pd.notna(row["perturb_duration"]) else -1,
    )


def load_summary_results(stress_dir):
    rows = []
    stress_path = Path(stress_dir)

    for path in sorted(stress_path.glob("results_*.json")):
        match = SUMMARY_RE.match(path.name)
        if not match:
            print(f"Skipping unrecognized result filename: {path.name}", flush=True)
            continue

        with path.open() as handle:
            data = json.load(handle)

        config = data.get("config", match.group("config"))
        perturbation = data.get("perturbation", match.group("perturbation"))
        family, amplitude, duration, label = parse_perturbation(perturbation)
        detail_path = stress_path / f"details_{config}_{perturbation}.tsv"

        row = {
            "config": config,
            "config_family": config_family(config),
            "noise_pct": noise_pct(config),
            "perturbation": perturbation,
            "perturb_family": family,
            "perturb_amplitude": amplitude,
            "perturb_duration": duration,
            "perturb_label": label,
            "n_samples": data.get("n_samples"),
            "wer": data.get("wer"),
            "wacc": data.get("wacc"),
            "mean_sample_wacc": data.get("mean_sample_wacc"),
            "avg_normalized_plausibility": data.get("avg_normalized_plausibility"),
            "avg_raw_plausibility": data.get("avg_raw_plausibility"),
            "hallucination_like_rate": data.get("hallucination_like_rate"),
            "mean_bleu": data.get("mean_bleu"),
            "bigram_repeats": data.get("sentences_with_bigram_repeats"),
            "trigram_repeats": data.get("sentences_with_trigram_repeats"),
            "fourgram_repeats": data.get("sentences_with_4gram_repeats"),
            "summary_path": str(path),
            "details_path": str(detail_path) if detail_path.exists() else "",
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No results_*.json files found in {stress_dir}")

    df = pd.DataFrame(rows)
    df["perturb_sort"] = df.apply(sort_key_for_perturbation, axis=1)
    df = df.sort_values(["perturb_sort", "config_family", "noise_pct", "config"]).reset_index(drop=True)
    return df.drop(columns=["perturb_sort"])


def load_detail_results(summary_df, max_points_per_group=250, random_seed=13):
    rng = np.random.default_rng(random_seed)
    frames = []

    for row in summary_df.itertuples(index=False):
        if not row.details_path:
            continue
        details = pd.read_csv(row.details_path, sep="\t")
        details["config"] = row.config
        details["config_family"] = row.config_family
        details["noise_pct"] = row.noise_pct
        details["perturbation"] = row.perturbation
        details["perturb_family"] = row.perturb_family
        details["perturb_amplitude"] = row.perturb_amplitude
        details["perturb_duration"] = row.perturb_duration
        details["perturb_label"] = row.perturb_label

        if max_points_per_group and len(details) > max_points_per_group:
            sampled_idx = rng.choice(details.index.to_numpy(), size=max_points_per_group, replace=False)
            details = details.loc[np.sort(sampled_idx)]

        frames.append(details)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def metric_label(metric):
    labels = {
        "wer": "WER",
        "wacc": "WAcc",
        "mean_bleu": "BLEU",
        "avg_normalized_plausibility": "GPT2 normalized plausibility",
        "bigram_repeats": "Bigram repeat count",
        "trigram_repeats": "Trigram repeat count",
        "fourgram_repeats": "4-gram repeat count",
        "delta_wer_vs_base": "Delta WER vs base",
        "delta_mean_bleu_vs_base": "Delta BLEU vs base",
        "delta_wacc_vs_base": "Delta WAcc vs base",
        "delta_fourgram_repeats_vs_base": "Delta 4-gram repeats vs base",
    }
    return labels.get(metric, metric.replace("_", " "))


def add_base_deltas(summary_df):
    df = summary_df.copy()
    base = df[df["config_family"] == "base"]
    metric_cols = ["wer", "wacc", "mean_bleu", "fourgram_repeats"]

    for metric in metric_cols:
        base_lookup = base.set_index("perturbation")[metric].to_dict()
        df[f"delta_{metric}_vs_base"] = df.apply(
            lambda row: row[metric] - base_lookup[row["perturbation"]]
            if row["perturbation"] in base_lookup and pd.notna(row[metric]) else np.nan,
            axis=1,
        )

    return df


def save_table(df, output_dir, name):
    out_path = Path(output_dir) / name
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}", flush=True)


def draw_heatmap(ax, matrix, title, cmap, center=None):
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap=cmap)

    if center is not None:
        max_abs = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        image.set_clim(-max_abs, max_abs)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", fontsize=7)

    return image


def plot_response_heatmaps(summary_df, output_dir):
    print("Generating Figure 2: response matrix heatmaps...", flush=True)
    df = add_base_deltas(summary_df)
    metrics = [
        "delta_wer_vs_base",
        "delta_mean_bleu_vs_base",
        "delta_fourgram_repeats_vs_base",
        "wer",
        "mean_bleu",
        "fourgram_repeats",
    ]

    plot_df = df[df["config_family"] != "base"].copy()
    if plot_df.empty:
        plot_df = df.copy()

    perturb_order = df.drop_duplicates("perturbation")["perturbation"].tolist()
    perturb_labels = df.drop_duplicates("perturbation").set_index("perturbation")["perturb_label"].to_dict()

    for metric in metrics:
        if metric not in plot_df.columns or plot_df[metric].notna().sum() == 0:
            continue

        matrix = plot_df.pivot_table(index="config", columns="perturbation", values=metric, aggfunc="first")
        matrix = matrix.reindex(columns=[p for p in perturb_order if p in matrix.columns])
        matrix = matrix.rename(columns=perturb_labels)

        fig_width = max(7, 1.25 * len(matrix.columns) + 2)
        fig_height = max(4, 0.45 * len(matrix.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        is_delta = metric.startswith("delta_")
        image = draw_heatmap(
            ax,
            matrix,
            f"Stress response matrix: {metric_label(metric)}",
            cmap="coolwarm" if is_delta else "viridis",
            center=0 if is_delta else None,
        )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(metric_label(metric))
        plt.tight_layout()
        save_figure(fig, output_dir, f"fig2_heatmap_{metric}")

    save_table(df, output_dir, "stress_response_matrix.csv")


def plot_dose_response(summary_df, output_dir):
    print("Generating Figure 3: perturbation dose-response curves...", flush=True)
    metrics = [("wer", "WER"), ("mean_bleu", "BLEU"), ("fourgram_repeats", "4-gram repeat count")]
    families = [family for family in ["full_noise", "onset_noise", "reverb"] if family in set(summary_df["perturb_family"])]

    if not families:
        print("No perturbation families with amplitudes found; skipping dose-response curves", flush=True)
        return

    fig, axes = plt.subplots(len(metrics), len(families), figsize=(5.5 * len(families), 4.0 * len(metrics)), squeeze=False)

    for col_idx, family in enumerate(families):
        family_df = summary_df[summary_df["perturb_family"] == family].copy()
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for config, config_df in family_df.groupby("config"):
                config_df = config_df.sort_values("perturb_amplitude")
                color = CONFIG_COLORS.get(config_family(config), "#525252")
                marker = CONFIG_MARKERS.get(config_family(config), "o")
                ax.plot(
                    config_df["perturb_amplitude"],
                    config_df[metric],
                    marker=marker,
                    linewidth=2,
                    markersize=7,
                    color=color,
                    label=config,
                )

            ax.set_title(f"{family}: {ylabel}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Perturbation amplitude")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, "fig3_dose_response")


def plot_wrong_but_fluent_scatter(detail_df, output_dir):
    print("Generating Figure 4: wrong-but-fluent scatter plots...", flush=True)
    if detail_df.empty:
        print("No detail TSV files found; skipping scatter plots", flush=True)
        return

    required = {"wacc", "norm_plausibility"}
    if not required.issubset(detail_df.columns):
        print(f"Missing columns {required - set(detail_df.columns)}; skipping scatter plots", flush=True)
        return

    focus = detail_df[detail_df["perturb_family"].isin(["full_noise", "onset_noise", "reverb"])].copy()
    if focus.empty:
        focus = detail_df.copy()

    perturbations = focus.drop_duplicates("perturbation").sort_values(
        ["perturb_family", "perturb_amplitude", "perturb_duration"]
    )["perturbation"].tolist()
    max_panels = min(6, len(perturbations))
    perturbations = perturbations[:max_panels]

    fig, axes = plt.subplots(1, max_panels, figsize=(5.0 * max_panels, 4.4), squeeze=False)

    for ax, perturbation in zip(axes[0], perturbations):
        sub = focus[focus["perturbation"] == perturbation]
        for config, config_df in sub.groupby("config"):
            color = CONFIG_COLORS.get(config_family(config), "#525252")
            marker = CONFIG_MARKERS.get(config_family(config), "o")
            ax.scatter(
                config_df["wacc"],
                config_df["norm_plausibility"],
                s=14,
                alpha=0.38,
                color=color,
                marker=marker,
                edgecolors="none",
                label=config,
            )

        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0.9, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(sub["perturb_label"].iloc[0], fontsize=10, fontweight="bold")
        ax.set_xlabel("WAcc")
        ax.set_xlim(-5.0, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.text(-4.85, 0.96, "wrong + fluent", fontsize=8, va="top")

    axes[0][0].set_ylabel("Normalized plausibility")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=9)
        fig.subplots_adjust(top=0.82)

    save_figure(fig, output_dir, "fig4_wrong_but_fluent_scatter")


def load_full_detail_results(summary_df):
    frames = []
    for row in summary_df.itertuples(index=False):
        if not row.details_path:
            continue
        details = pd.read_csv(row.details_path, sep="\t", usecols=lambda col: col in {"wer", "wacc", "norm_plausibility"})
        details["config"] = row.config
        details["config_family"] = row.config_family
        details["noise_pct"] = row.noise_pct
        details["perturbation"] = row.perturbation
        details["perturb_label"] = row.perturb_label
        frames.append(details)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_catastrophic_tail(summary_df, output_dir):
    print("Generating Figure 5: catastrophic error tail plots...", flush=True)
    detail_df = load_full_detail_results(summary_df)
    if detail_df.empty or "wer" not in detail_df.columns:
        print("No detail WER data found; skipping catastrophic tail plots", flush=True)
        return

    thresholds = [1, 2, 5, 10, 20]
    rows = []
    grouped = detail_df.groupby(["config", "config_family", "noise_pct", "perturbation", "perturb_label"], dropna=False)
    for group_key, group_df in grouped:
        config, family, pct, perturbation, label = group_key
        row = {
            "config": config,
            "config_family": family,
            "noise_pct": pct,
            "perturbation": perturbation,
            "perturb_label": label,
            "n_samples": len(group_df),
        }
        for threshold in thresholds:
            row[f"wer_gt_{threshold}"] = float((group_df["wer"] > threshold).mean())
        row["wrong_fluent_rate"] = float(((group_df["wacc"] < 0) & (group_df["norm_plausibility"] > 0.9)).mean())
        rows.append(row)

    tail_df = pd.DataFrame(rows).sort_values(["perturbation", "config_family", "noise_pct", "config"])
    save_table(tail_df, output_dir, "catastrophic_tail_rates.csv")

    selected_perturbations = tail_df[tail_df["perturbation"].str.contains("full_noise|reverb", regex=True)]["perturbation"].unique().tolist()
    if not selected_perturbations:
        selected_perturbations = tail_df["perturbation"].unique().tolist()
    selected_perturbations = selected_perturbations[:4]

    fig, axes = plt.subplots(1, len(selected_perturbations), figsize=(5.2 * len(selected_perturbations), 4.4), squeeze=False)

    for ax, perturbation in zip(axes[0], selected_perturbations):
        sub = tail_df[tail_df["perturbation"] == perturbation]
        for _, row in sub.iterrows():
            y_values = [row[f"wer_gt_{threshold}"] for threshold in thresholds]
            color = CONFIG_COLORS.get(row["config_family"], "#525252")
            marker = CONFIG_MARKERS.get(row["config_family"], "o")
            ax.plot(thresholds, y_values, color=color, marker=marker, linewidth=2, label=row["config"])

        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("WER threshold")
        ax.set_ylabel("Fraction of utterances above threshold")
        ax.set_title(sub["perturb_label"].iloc[0], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_dir, "fig5_catastrophic_tail")


def save_figure(fig, output_dir, stem):
    output_path = Path(output_dir)
    png_path = output_path / f"{stem}.png"
    pdf_path = output_path / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)
    print(f"Saved {pdf_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress_dir", required=True, help="Directory containing results_*.json and details_*.tsv files")
    parser.add_argument("--output_dir", required=True, help="Directory for plots and derived CSVs")
    parser.add_argument("--scatter_points_per_group", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_df = load_summary_results(args.stress_dir)
    save_table(summary_df, args.output_dir, "stress_summary_long.csv")
    print(f"Loaded {len(summary_df)} summary rows", flush=True)

    detail_sample_df = load_detail_results(summary_df, max_points_per_group=args.scatter_points_per_group)
    if not detail_sample_df.empty:
        save_table(detail_sample_df, args.output_dir, "stress_details_scatter_sample.csv")

    plot_response_heatmaps(summary_df, args.output_dir)
    plot_dose_response(summary_df, args.output_dir)
    plot_wrong_but_fluent_scatter(detail_sample_df, args.output_dir)
    plot_catastrophic_tail(summary_df, args.output_dir)


if __name__ == "__main__":
    main()