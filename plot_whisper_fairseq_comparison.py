"""
Create a two-panel Whisper vs fairseq comparison figure.

Panel A: grouped WER bars by condition.
Panel B: baseline-relative WER-vs-fluency or WER-vs-repetition scatter.

The default inputs use the 64% Whisper validation outputs and the fairseq
aggregate table produced by evaluate_fairseq_results.py.
"""

import argparse
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


CONDITIONS = ["base", "RR", "RU", "UR", "UU"]
CONDITION_LABELS = {"base": "Base", "RR": "RR", "RU": "RU", "UR": "UR", "UU": "UU"}
CONDITION_COLORS = {
    "base": "#4D4D4D",
    "RR": "#0072B2",
    "RU": "#E69F00",
    "UR": "#CC79A7",
    "UU": "#009E73",
}
FAMILY_MARKERS = {"Whisper 64%": "o", "fairseq 8%": "^"}
FAMILY_COLORS = {"Whisper 64%": "#7B3294", "fairseq 8%": "#E66101"}
FAMILY_HATCHES = {"Whisper 64%": "", "fairseq 8%": "///"}
SCATTER_X_DODGE = {"Whisper 64%": -0.004, "fairseq 8%": 0.004}
SCATTER_Y_DODGE = {"Whisper 64%": 0.0015, "fairseq 8%": -0.0015}
PATTERN_X_DODGE = {"Whisper 64%": -0.0025, "fairseq 8%": 0.0025}
PATTERN_Y_DODGE = {"Whisper 64%": 0.0009, "fairseq 8%": -0.0009}
LABEL_OFFSETS = {
    ("Whisper 64%", "RR"): (10, 12),
    ("Whisper 64%", "RU"): (-10, 12),
    ("Whisper 64%", "UR"): (10, -10),
    ("Whisper 64%", "UU"): (12, -10),
    ("fairseq 8%", "RR"): (10, -12),
    ("fairseq 8%", "RU"): (-10, -12),
    ("fairseq 8%", "UR"): (-10, 10),
    ("fairseq 8%", "UU"): (12, 10),
}
REPETITION_LABEL_OFFSETS = {
    ("Whisper 64%", "RR"): (10, 12),
    ("Whisper 64%", "RU"): (12, 10),
    ("Whisper 64%", "UR"): (10, -12),
    ("Whisper 64%", "UU"): (12, -10),
    ("fairseq 8%", "RR"): (10, -10),
    ("fairseq 8%", "RU"): (-10, -10),
    ("fairseq 8%", "UR"): (-12, 10),
    ("fairseq 8%", "UU"): (12, 10),
}
CROWDED_LABELS = {
    "fluency_avg": {
        ("Whisper 64%", "RR"),
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UU"),
        ("fairseq 8%", "RU"),
    },
    "repetition": {
        ("Whisper 64%", "RR"),
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UU"),
    },
    "repetition_fourgram": {
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UU"),
    },
    "repetition_avg": {
        ("Whisper 64%", "RR"),
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UU"),
    },
}


DEFAULT_WHISPER_FILES = {
    "base": ["/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/per_utterance_base_ckpt14000.csv"],
    "RR": ["/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_rr_64pct_checkpoint-9375.csv"],
    "RU": ["/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_ru_64pct_checkpoint-9375.csv"],
    "UR": [
        "/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_ur_64pct_checkpoint-10000_shard00-of-02.csv",
        "/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_ur_64pct_checkpoint-10000_shard01-of-02.csv",
    ],
    "UU": [
        "/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_uu_64pct_final_shard00-of-02.csv",
        "/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/per_utterance_uu_64pct_final_shard01-of-02.csv",
    ],
}


def load_per_utterance(paths):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def aggregate_whisper(whisper_files):
    rows = []
    for condition in CONDITIONS:
        df = load_per_utterance(whisper_files[condition])
        row = {
            "family": "Whisper 64%",
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n_samples": len(df),
            "mean_wer": df["wer"].mean(),
            "mean_wacc": df["wacc"].mean(),
            "mean_cosine_similarity": df["cosine_similarity"].mean(),
            "mean_bleu": df["bleu"].mean(),
            "mean_bigram_rep_count": df["bigram_rep_count"].mean(),
            "mean_trigram_rep_count": df["trigram_rep_count"].mean(),
            "mean_fourgram_rep_count": df["fourgram_rep_count"].mean(),
        }
        fluency_cols = []
        if "normalized_sentence_score_gpt2" in df.columns:
            row["fluency_gpt2"] = df["normalized_sentence_score_gpt2"].mean()
            fluency_cols.append("fluency_gpt2")
        if "normalized_sentence_score_Qwen3-0.6B" in df.columns:
            row["fluency_qwen3_0_6b"] = df["normalized_sentence_score_Qwen3-0.6B"].mean()
            fluency_cols.append("fluency_qwen3_0_6b")
        if not fluency_cols:
            raise ValueError(f"No normalized fluency column for {condition}")
        row["fluency"] = float(np.mean([row[col] for col in fluency_cols]))
        row["fluency_source"] = "+".join(fluency_cols)
        row["mean_repetition_count"] = float(np.mean([
            row["mean_bigram_rep_count"],
            row["mean_trigram_rep_count"],
            row["mean_fourgram_rep_count"],
        ]))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_fairseq(path):
    df = pd.read_csv(path)
    rows = []
    for condition in CONDITIONS:
        lookup = condition.lower()
        cond_df = df[df["condition"].str.lower() == lookup]
        if len(cond_df) != 1:
            raise ValueError(f"Expected one fairseq row for {condition}, found {len(cond_df)}")
        source = cond_df.iloc[0]
        fluency_values = []
        fluency_source = []
        if "mean_normalized_score_gpt2" in source.index:
            fluency_values.append(source["mean_normalized_score_gpt2"])
            fluency_source.append("gpt2")
        if "mean_normalized_score_Qwen3-0.6B" in source.index:
            fluency_values.append(source["mean_normalized_score_Qwen3-0.6B"])
            fluency_source.append("Qwen3-0.6B")
        if not fluency_values:
            fluency_values.append(source["primary_fluency_mean"])
            fluency_source.append("fairseq scaled probability")
        fluency = float(np.mean(fluency_values))
        mean_repetition_count = float(np.mean([
            source["mean_bigram_rep_count"],
            source["mean_trigram_rep_count"],
            source["mean_fourgram_rep_count"],
        ]))
        rows.append({
            "family": "fairseq 8%",
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n_samples": int(source["n_samples"]),
            "mean_wer": source["mean_wer"],
            "mean_wacc": source["mean_wacc"],
            "mean_cosine_similarity": source["mean_cosine_similarity"],
            "mean_bleu": source["mean_bleu"],
            "mean_bigram_rep_count": source["mean_bigram_rep_count"],
            "mean_trigram_rep_count": source["mean_trigram_rep_count"],
            "mean_fourgram_rep_count": source["mean_fourgram_rep_count"],
            "mean_repetition_count": mean_repetition_count,
            "fluency": fluency,
            "fluency_source": "+".join(fluency_source),
            "fluency_gpt2": source.get("mean_normalized_score_gpt2", np.nan),
            "fluency_qwen3_0_6b": source.get("mean_normalized_score_Qwen3-0.6B", np.nan),
        })
    return pd.DataFrame(rows)


def add_baseline_deltas(plot_df):
    plot_df = plot_df.copy()
    plot_df["delta_wer"] = np.nan
    plot_df["delta_wacc"] = np.nan
    plot_df["delta_fluency"] = np.nan
    plot_df["delta_trigram_rep_count"] = np.nan
    plot_df["delta_fourgram_rep_count"] = np.nan
    plot_df["delta_mean_repetition_count"] = np.nan
    plot_df["relative_wer"] = np.nan
    plot_df["relative_fluency"] = np.nan
    for family in plot_df["family"].unique():
        family_mask = plot_df["family"] == family
        base_row = plot_df[family_mask & (plot_df["condition"].astype(str) == "base")]
        if len(base_row) != 1:
            raise ValueError(f"Expected one baseline row for {family}, found {len(base_row)}")
        base_wacc = base_row.iloc[0]["mean_wacc"]
        base_wer = base_row.iloc[0]["mean_wer"]
        base_fluency = base_row.iloc[0]["fluency"]
        base_trigram_rep = base_row.iloc[0]["mean_trigram_rep_count"]
        base_fourgram_rep = base_row.iloc[0]["mean_fourgram_rep_count"]
        base_mean_rep = base_row.iloc[0]["mean_repetition_count"]
        plot_df.loc[family_mask, "delta_wer"] = plot_df.loc[family_mask, "mean_wer"] - base_wer
        plot_df.loc[family_mask, "delta_wacc"] = plot_df.loc[family_mask, "mean_wacc"] - base_wacc
        plot_df.loc[family_mask, "delta_fluency"] = plot_df.loc[family_mask, "fluency"] - base_fluency
        plot_df.loc[family_mask, "relative_wer"] = plot_df.loc[family_mask, "mean_wer"] / base_wer
        plot_df.loc[family_mask, "relative_fluency"] = plot_df.loc[family_mask, "fluency"] / base_fluency
        plot_df.loc[family_mask, "delta_trigram_rep_count"] = (
            plot_df.loc[family_mask, "mean_trigram_rep_count"] - base_trigram_rep
        )
        plot_df.loc[family_mask, "delta_fourgram_rep_count"] = (
            plot_df.loc[family_mask, "mean_fourgram_rep_count"] - base_fourgram_rep
        )
        plot_df.loc[family_mask, "delta_mean_repetition_count"] = (
            plot_df.loc[family_mask, "mean_repetition_count"] - base_mean_rep
        )
    return plot_df


def annotate_condition(ax, row, x_value, y_value, crowded_labels, offset_map=None):
    if str(row["condition"]) == "base":
        return
    label = f"{'Whisper' if row['family'] == 'Whisper 64%' else 'fairseq'} {row['condition_label']}"
    label_key = (row["family"], str(row["condition"]))
    if offset_map and label_key in offset_map:
        x_offset, y_offset = offset_map[label_key]
    else:
        x_offset, y_offset = (5, 5)
    arrowprops = None
    if label_key in crowded_labels:
        arrowprops = {
            "arrowstyle": "-",
            "color": "#555555",
            "lw": 0.55,
            "shrinkA": 2,
            "shrinkB": 4,
            "alpha": 0.8,
        }
    ax.annotate(
        label,
        (x_value, y_value),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        fontsize=8.0,
        ha="left" if x_offset >= 0 else "right",
        va="center",
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
        arrowprops=arrowprops,
    )


def make_pattern_figure(plot_df, output_dir):
    families = ["Whisper 64%", "fairseq 8%"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax_rep, ax_hall = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(axis="both", labelsize=10)

    rep_offsets = {
        ("Whisper 64%", "RR"): (5, 7),
        ("Whisper 64%", "RU"): (5, 5),
        ("Whisper 64%", "UR"): (5, -7),
        ("Whisper 64%", "UU"): (5, -5),
        ("fairseq 8%", "RR"): (5, 6),
        ("fairseq 8%", "RU"): (-5, -5),
        ("fairseq 8%", "UR"): (5, 5),
        ("fairseq 8%", "UU"): (-5, -5),
    }
    hall_offsets = {
        ("Whisper 64%", "RR"): (5, -6),
        ("Whisper 64%", "RU"): (5, 5),
        ("Whisper 64%", "UR"): (-8, -12),
        ("Whisper 64%", "UU"): (-5, -6),
        ("fairseq 8%", "RR"): (5, -6),
        ("fairseq 8%", "RU"): (-5, -5),
        ("fairseq 8%", "UR"): (-8, 16),
        ("fairseq 8%", "UU"): (-5, -6),
    }
    rep_crowded = {
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UU"),
        ("fairseq 8%", "RU"),
        ("fairseq 8%", "UR"),
    }
    hall_crowded = {
        ("Whisper 64%", "RR"),
        ("Whisper 64%", "RU"),
        ("Whisper 64%", "UR"),
        ("Whisper 64%", "UU"),
        ("fairseq 8%", "RU"),
        ("fairseq 8%", "UR"),
        ("fairseq 8%", "UU"),
    }

    for _, row in plot_df.iterrows():
        x_dodge = PATTERN_X_DODGE[row["family"]]
        y_dodge = PATTERN_Y_DODGE[row["family"]]

        rep_x = row["delta_wer"] + x_dodge
        rep_y = row["delta_fourgram_rep_count"] + y_dodge
        ax_rep.scatter(
            rep_x,
            rep_y,
            s=135,
            marker=FAMILY_MARKERS[row["family"]],
            color=CONDITION_COLORS[row["condition"]],
            edgecolor="black",
            linewidth=0.65,
            alpha=0.92,
        )
        annotate_condition(ax_rep, row, rep_x, rep_y, rep_crowded, rep_offsets)

        hall_x = row["mean_wer"] + x_dodge
        hall_y = row["fluency"] + y_dodge
        ax_hall.scatter(
            hall_x,
            hall_y,
            s=135,
            marker=FAMILY_MARKERS[row["family"]],
            color=CONDITION_COLORS[row["condition"]],
            edgecolor="black",
            linewidth=0.65,
            alpha=0.92,
        )
        annotate_condition(ax_hall, row, hall_x, hall_y, hall_crowded, hall_offsets)

    ax_rep.axhspan(0.020, 0.075, color="#E8D8F0", alpha=0.45, zorder=0)
    ax_rep.text(
        0.255,
        0.066,
        "high repetition",
        fontsize=9,
        color="#5E3C67",
        ha="right",
        va="center",
    )
    ax_rep.axhline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax_rep.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax_rep.set_title("A. Repetition Pattern", fontsize=14, fontweight="bold")
    ax_rep.set_xlabel("ΔWER vs own base", fontsize=12)
    ax_rep.set_ylabel("Δ4-gram repetitions / own base", fontsize=12)
    ax_rep.set_xlim(-0.02, 0.32)
    ax_rep.set_ylim(-0.16, 0.075)
    ax_rep.grid(True, alpha=0.2, linewidth=0.8)

    ax_hall.axhspan(0.75, 1.0, color="#F6E3C3", alpha=0.55, zorder=0)
    ax_hall.axvspan(0.32, 0.46, color="#DCE7F2", alpha=0.14, zorder=0)
    ax_hall.axvspan(
        0.32,
        0.46,
        facecolor="none",
        edgecolor="#A9B8C8",
        hatch="//",
        linewidth=0.0,
        zorder=0,
    )
    ax_hall.text(
        0.300,
        0.965,
        "fluent plausible sentences",
        fontsize=9,
        color="#8A5A00",
        ha="right",
        va="center",
    )
    ax_hall.text(
        0.445,
        0.115,
        "transcription failure",
        fontsize=9,
        color="#48596D",
        ha="right",
        va="center",
    )
    ax_hall.axhline(0.75, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax_hall.set_title("B. Hallucination-like Pattern", fontsize=14, fontweight="bold")
    ax_hall.set_xlabel("WER", fontsize=12)
    ax_hall.set_ylabel("Fluency", fontsize=12)
    ax_hall.set_xlim(0.0, 0.46)
    ax_hall.set_ylim(0.0, 1.0)
    ax_hall.grid(True, alpha=0.2, linewidth=0.8)

    model_handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[family], color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=9, linestyle="None", label=family)
        for family in families
    ]
    condition_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CONDITION_COLORS[condition],
               markeredgecolor="black", markersize=9, linestyle="None", label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    fig.legend(
        handles=model_handles,
        title="Model",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
        loc="upper right",
        bbox_to_anchor=(0.970, 0.86),
        ncol=1,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=condition_handles,
        title="Condition",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
        loc="upper right",
        bbox_to_anchor=(0.970, 0.60),
        ncol=1,
        borderaxespad=0.0,
    )

    fig.suptitle("Repetition and Hallucination-like Failure Patterns", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.915, 0.95])

    output_stem = "whisper_fairseq_repetition_hallucination_patterns"
    png_path = os.path.join(output_dir, f"{output_stem}.png")
    pdf_path = os.path.join(output_dir, f"{output_stem}.pdf")
    jpg_path = os.path.join(output_dir, f"{output_stem}.jpg")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    buffer.seek(0)
    rendered = Image.open(buffer).convert("RGBA")
    white = Image.new("RGBA", rendered.size, "white")
    white.alpha_composite(rendered)
    flattened = white.convert("RGB")
    flattened.save(png_path)
    flattened.save(jpg_path, quality=95)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return png_path, pdf_path, jpg_path


def get_y_config(y_metric):
    if y_metric == "fluency_avg":
        return {
            "column": "delta_fluency",
            "ylabel": "ΔMean fluency (GPT-2 + Qwen3-0.6B) vs own base",
            "ylim": (-0.065, 0.03),
            "vertical_label": "lower fluency",
            "vertical_xy": (0.315, -0.043),
            "vertical_text": (0.315, -0.006),
            "suffix": "",
        }
    if y_metric == "repetition":
        return {
            "column": "delta_trigram_rep_count",
            "ylabel": "ΔTrigram repetition count vs own base",
            "ylim": (-0.18, 0.12),
            "vertical_label": "more repetition",
            "vertical_xy": (0.315, 0.095),
            "vertical_text": (0.315, 0.020),
            "suffix": "_repetition",
        }
    if y_metric == "repetition_fourgram":
        return {
            "column": "delta_fourgram_rep_count",
            "ylabel": "Δ4-gram repetition count vs own base",
            "ylim": (-0.18, 0.12),
            "vertical_label": "more 4-gram repetition",
            "vertical_xy": (0.315, 0.095),
            "vertical_text": (0.315, 0.020),
            "suffix": "_fourgram_repetition",
        }
    if y_metric == "repetition_avg":
        return {
            "column": "delta_mean_repetition_count",
            "ylabel": "ΔMean 2/3/4-gram repetition count vs own base",
            "ylim": (-0.18, 0.12),
            "vertical_label": "more repetition",
            "vertical_xy": (0.315, 0.095),
            "vertical_text": (0.315, 0.020),
            "suffix": "_avg_repetition",
        }
    raise ValueError(f"Unsupported y_metric: {y_metric}")


def make_figure(plot_df, output_dir, y_metric="fluency"):
    y_config = get_y_config(y_metric)
    families = ["Whisper 64%", "fairseq 8%"]
    x = np.arange(len(CONDITIONS))
    width = 0.34

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax_bar, ax_scatter = axes
    for ax in axes:
        ax.set_facecolor("white")

    offsets = {"Whisper 64%": -width / 2, "fairseq 8%": width / 2}

    for family in families:
        for condition in CONDITIONS:
            row = plot_df[(plot_df["family"] == family) & (plot_df["condition"] == condition)].iloc[0]
            ax_bar.bar(
                x[CONDITIONS.index(condition)] + offsets[family],
                row["mean_wer"],
                width,
                label=family,
                color=CONDITION_COLORS[condition],
                edgecolor="black",
                linewidth=0.6,
                hatch=FAMILY_HATCHES[family],
            )

    ax_bar.set_title("A. Recognition Error", fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("WER", fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax_bar.set_ylim(0, 0.46)
    ax_bar.grid(True, axis="y", alpha=0.2, linewidth=0.8)
    ax_bar.legend(
        handles=[
             Patch(facecolor="#D9D9D9", edgecolor="black", label="Whisper 64%"),
             Patch(facecolor="#D9D9D9", edgecolor="black", hatch="///", label="fairseq 8%"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )

    for _, row in plot_df.iterrows():
        x_offset = SCATTER_X_DODGE[row["family"]]
        y_offset = SCATTER_Y_DODGE[row["family"]]
        ax_scatter.scatter(
            row["delta_wer"] + x_offset,
            row[y_config["column"]] + y_offset,
            s=115,
            marker=FAMILY_MARKERS[row["family"]],
            color=CONDITION_COLORS[row["condition"]],
            edgecolor="black",
            linewidth=0.65,
            alpha=0.92,
        )
        if str(row["condition"]) != "base":
            label = f"{'Whisper' if row['family'] == 'Whisper 64%' else 'fairseq'} {row['condition_label']}"
            label_offsets = REPETITION_LABEL_OFFSETS if y_metric.startswith("repetition") else LABEL_OFFSETS
            label_key = (row["family"], str(row["condition"]))
            label_x_offset, label_y_offset = label_offsets[label_key]
            arrowprops = None
            if label_key in CROWDED_LABELS.get(y_metric, set()):
                arrowprops = {
                    "arrowstyle": "-",
                    "color": "#555555",
                    "lw": 0.55,
                    "shrinkA": 2,
                    "shrinkB": 4,
                    "alpha": 0.8,
                }
            ax_scatter.annotate(
                label,
                (row["delta_wer"] + x_offset, row[y_config["column"]] + y_offset),
                xytext=(label_x_offset, label_y_offset),
                textcoords="offset points",
                fontsize=7.4,
                ha="left" if label_x_offset > 0 else "right",
                va="center",
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                arrowprops=arrowprops,
            )

    ax_scatter.axhline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    ax_scatter.axvline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    ax_scatter.set_title("B. Baseline-Relative Failure Space", fontsize=13, fontweight="bold")
    ax_scatter.set_xlabel("ΔWER vs own base", fontsize=11)
    ax_scatter.set_ylabel(y_config["ylabel"], fontsize=11)
    ax_scatter.set_xlim(-0.02, 0.34)
    ax_scatter.set_ylim(*y_config["ylim"])
    ax_scatter.grid(True, alpha=0.2, linewidth=0.8)

    ax_scatter.annotate(
        "more lexical\ndegeneration",
        xy=(0.115, y_config["ylim"][1] - 0.006),
        xytext=(0.025, y_config["ylim"][1] - 0.006),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8,
        ha="left",
        va="center",
        color="#555555",
    )
    ax_scatter.annotate(
        y_config["vertical_label"],
        xy=y_config["vertical_xy"],
        xytext=y_config["vertical_text"],
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8,
        ha="center",
        va="center",
        rotation=90,
        color="#555555",
    )

    model_handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[family], color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, linestyle="None", label=family)
        for family in families
    ]
    condition_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CONDITION_COLORS[condition],
               markeredgecolor="black", markersize=8, linestyle="None", label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    model_legend = ax_scatter.legend(
        handles=model_handles,
        title="Model",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    ax_scatter.add_artist(model_legend)
    ax_scatter.legend(
        handles=condition_handles,
        title="Condition",
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.68),
        borderaxespad=0,
    )

    fig.suptitle("Whisper and fairseq Failure Modes by Training-Noise Condition", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])

    output_stem = f"whisper_fairseq_two_panel{y_config['suffix']}"
    png_path = os.path.join(output_dir, f"{output_stem}.png")
    pdf_path = os.path.join(output_dir, f"{output_stem}.pdf")
    jpg_path = os.path.join(output_dir, f"{output_stem}.jpg")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    buffer.seek(0)
    rendered = Image.open(buffer).convert("RGBA")
    white = Image.new("RGBA", rendered.size, "white")
    white.alpha_composite(rendered)
    flattened = white.convert("RGB")
    flattened.save(png_path)
    flattened.save(jpg_path, quality=95)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return png_path, pdf_path, jpg_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairseq_aggregate", default="fairseq_eval_lm/aggregate_metrics_fairseq.csv")
    parser.add_argument("--output_dir", default="plots")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    whisper_df = aggregate_whisper(DEFAULT_WHISPER_FILES)
    fairseq_df = aggregate_fairseq(args.fairseq_aggregate)
    plot_df = pd.concat([whisper_df, fairseq_df], ignore_index=True)
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=CONDITIONS, ordered=True)
    plot_df = plot_df.sort_values(["family", "condition"])
    plot_df = add_baseline_deltas(plot_df)

    csv_path = os.path.join(args.output_dir, "whisper_fairseq_two_panel_data.csv")
    plot_df.to_csv(csv_path, index=False)
    repetition_csv_path = os.path.join(args.output_dir, "whisper_fairseq_two_panel_repetition_data.csv")
    plot_df.to_csv(repetition_csv_path, index=False)
    png_path, pdf_path, jpg_path = make_figure(plot_df, args.output_dir, y_metric="fluency_avg")
    rep_png_path, rep_pdf_path, rep_jpg_path = make_figure(plot_df, args.output_dir, y_metric="repetition")
    four_png_path, four_pdf_path, four_jpg_path = make_figure(plot_df, args.output_dir, y_metric="repetition_fourgram")
    avg_rep_png_path, avg_rep_pdf_path, avg_rep_jpg_path = make_figure(plot_df, args.output_dir, y_metric="repetition_avg")
    pattern_png_path, pattern_pdf_path, pattern_jpg_path = make_pattern_figure(plot_df, args.output_dir)

    print(f"Saved data: {csv_path}", flush=True)
    print(f"Saved data: {repetition_csv_path}", flush=True)
    print(f"Saved figure: {png_path}", flush=True)
    print(f"Saved figure: {pdf_path}", flush=True)
    print(f"Saved figure: {jpg_path}", flush=True)
    print(f"Saved figure: {rep_png_path}", flush=True)
    print(f"Saved figure: {rep_pdf_path}", flush=True)
    print(f"Saved figure: {rep_jpg_path}", flush=True)
    print(f"Saved figure: {four_png_path}", flush=True)
    print(f"Saved figure: {four_pdf_path}", flush=True)
    print(f"Saved figure: {four_jpg_path}", flush=True)
    print(f"Saved figure: {avg_rep_png_path}", flush=True)
    print(f"Saved figure: {avg_rep_pdf_path}", flush=True)
    print(f"Saved figure: {avg_rep_jpg_path}", flush=True)
    print(f"Saved figure: {pattern_png_path}", flush=True)
    print(f"Saved figure: {pattern_pdf_path}", flush=True)
    print(f"Saved figure: {pattern_jpg_path}", flush=True)


if __name__ == "__main__":
    main()