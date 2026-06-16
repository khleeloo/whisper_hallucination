"""
Aggregation and analysis for cross-model validation experiments.

Reads per-utterance CSV files from evaluate_whisper_validation.py and produces:
  - aggregate_metrics_whisper.csv
  - baseline_relative_deltas_whisper.csv
  - fluency_scorer_robustness.csv
  - copied_label_analysis.csv
  - qualitative_examples.csv
  - cross_model_comparison.csv (scaffold)

Usage:
    python analyze_validation.py \
        --per_utterance_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation \
        --noisy_labels_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --- Text normalization (same as eval script) ---
WHISPER_SPECIAL = re.compile(r"<\|[^|]+\|>")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = WHISPER_SPECIAL.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Load per-utterance data ---


def load_all_per_utterance(per_utterance_dir):
    """Load and merge all per-utterance CSV files."""
    csv_files = sorted([
        f for f in os.listdir(per_utterance_dir)
        if f.startswith("per_utterance_") and f.endswith(".csv")
    ])

    if not csv_files:
        raise FileNotFoundError(f"No per_utterance_*.csv files found in {per_utterance_dir}")

    dfs = []
    for f in csv_files:
        path = os.path.join(per_utterance_dir, f)
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows", flush=True)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(merged)} rows, {len(merged['model_name'].unique())} models", flush=True)
    return merged


# --- Aggregate metrics ---


def compute_aggregate_metrics(df):
    """Compute per evaluated model aggregate metrics."""
    # Identify LM columns
    lm_cols = [c for c in df.columns if c.startswith("normalized_sentence_score_")]
    lm_names = [c.replace("normalized_sentence_score_", "") for c in lm_cols]

    # Detect which LMs are weak vs strong based on naming
    weak_lms = [n for n in lm_names if "gpt2" in n.lower() or "small" in n.lower()]
    strong_lms = [n for n in lm_names if n not in weak_lms]

    if not weak_lms:
        weak_lms = lm_names[:1] if lm_names else ["gpt2"]
    if not strong_lms:
        strong_lms = lm_names[-1:] if lm_names else ["Qwen3-1.7B"]

    print(f"  Weak LMs: {weak_lms}", flush=True)
    print(f"  Strong LMs: {strong_lms}", flush=True)

    # Primary LM for key metrics
    primary_lm = strong_lms[0] if strong_lms else (weak_lms[0] if weak_lms else "gpt2")
    primary_norm_col = f"normalized_sentence_score_{primary_lm}"

    # Group by evaluated model, not only noise_condition. Otherwise rr/rr_32/rr_64pct
    # collapse into one row when their CSVs share noise_condition="rr".
    model_names = sorted(df["model_name"].unique())
    print(f"  Models: {model_names}", flush=True)

    agg_rows = []
    for model_name in model_names:
        cond_df = df[df["model_name"] == model_name]
        noise_conditions = sorted(cond_df["noise_condition"].dropna().unique())
        noise_ratios = sorted(cond_df["noise_ratio"].dropna().unique())
        row = {
            "condition": model_name,
            "model_name": model_name,
            "noise_condition": noise_conditions[0] if len(noise_conditions) == 1 else ";".join(map(str, noise_conditions)),
            "noise_ratio": noise_ratios[0] if len(noise_ratios) == 1 else ";".join(map(str, noise_ratios)),
            "n_samples": len(cond_df),
        }

        # WER/WAcc
        row["mean_wer"] = cond_df["wer"].mean()
        row["mean_wacc"] = cond_df["wacc"].mean()

        # Cosine
        row["mean_cosine_similarity"] = cond_df["cosine_similarity"].mean()

        # Fluency (per LM)
        for lm in lm_names:
            col = f"normalized_sentence_score_{lm}"
            if col in cond_df.columns:
                row[f"mean_normalized_score_{lm}"] = cond_df[col].mean()

        # Repetition rates
        row["mean_bigram_rep_count"] = cond_df["bigram_rep_count"].mean()
        row["mean_trigram_rep_count"] = cond_df["trigram_rep_count"].mean()
        row["mean_fourgram_rep_count"] = cond_df["fourgram_rep_count"].mean()

        # Repetition rates (proportion of utterances with repetition)
        row["bigram_rep_rate"] = cond_df["has_bigram_rep"].mean()
        row["trigram_rep_rate"] = cond_df["has_trigram_rep"].mean()
        row["fourgram_rep_rate"] = cond_df["has_fourgram_rep"].mean()

        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    return agg_df, weak_lms, strong_lms, primary_lm, primary_norm_col


# --- Hallucination-like rate ---


def compute_hallucination_rates(df, agg_df, primary_lm, primary_norm_col):
    """Compute hallucination-like rates using baseline-derived thresholds."""
    # Get baseline stats
    base_df = df[df["noise_condition"] == "base"]
    if len(base_df) == 0:
        print("  WARNING: No 'base' condition found. Using global means as thresholds.")
        wacc_threshold = df["wacc"].mean()
        fluency_threshold = df[primary_norm_col].mean() if primary_norm_col in df.columns else 0.5
        wacc_q25 = df["wacc"].quantile(0.25)
        fluency_median = df[primary_norm_col].median() if primary_norm_col in df.columns else 0.5
    else:
        wacc_threshold = base_df["wacc"].mean()
        fluency_threshold = base_df[primary_norm_col].mean() if primary_norm_col in base_df.columns else 0.5
        wacc_q25 = base_df["wacc"].quantile(0.25)
        fluency_median = base_df[primary_norm_col].median() if primary_norm_col in base_df.columns else 0.5

    print(f"  WAcc threshold (mean base): {wacc_threshold:.4f}", flush=True)
    print(f"  Fluency threshold (mean base): {fluency_threshold:.4f}", flush=True)
    print(f"  WAcc Q25 (strict): {wacc_q25:.4f}", flush=True)
    print(f"  Fluency median (strict): {fluency_median:.4f}", flush=True)

    # Compute per-model hallucination rates
    hall_rates = {}
    for cond in sorted(df["model_name"].unique()):
        cond_mask = df["model_name"] == cond
        cond_df = df[cond_mask]

        # Standard definition
        hall_standard = (
            (cond_df["wacc"] < wacc_threshold)
            & (cond_df[primary_norm_col] > fluency_threshold)
        )
        hall_rate_standard = hall_standard.mean()

        # Strict definition
        hall_strict = (
            (cond_df["wacc"] <= wacc_q25)
            & (cond_df[primary_norm_col] >= fluency_median)
        )
        hall_rate_strict = hall_strict.mean()

        hall_rates[cond] = {
            "hallucination_like_rate": hall_rate_standard,
            "hallucination_like_rate_strict": hall_rate_strict,
            "wacc_threshold": wacc_threshold,
            "fluency_threshold": fluency_threshold,
        }

        # Also add hallucination flag to the main dataframe
        df.loc[cond_mask, "hallucination_like"] = hall_standard.values
        df.loc[cond_mask, "hallucination_like_strict"] = hall_strict.values

    # Add to aggregate
    for row_idx in range(len(agg_df)):
        cond = agg_df.iloc[row_idx]["condition"]
        if cond in hall_rates:
            agg_df.at[row_idx, "hallucination_like_rate"] = hall_rates[cond]["hallucination_like_rate"]
            agg_df.at[row_idx, "hallucination_like_rate_strict"] = hall_rates[cond]["hallucination_like_rate_strict"]

    return df, agg_df, wacc_threshold, fluency_threshold


# --- Relative deltas ---


def compute_baseline_deltas(agg_df):
    """Compute relative changes from baseline."""
    base_row = agg_df[agg_df["condition"] == "base"]
    if len(base_row) == 0:
        print("  WARNING: No baseline row found. Skipping delta computation.")
        return agg_df

    base = base_row.iloc[0]
    delta_rows = []

    for _, row in agg_df.iterrows():
        drow = {"condition": row["condition"]}
        for col in agg_df.columns:
            if col in {"condition", "model_name", "noise_condition", "noise_ratio", "n_samples"}:
                drow[col] = row[col]
                continue
            base_val = base[col]
            curr_val = row[col]
            if pd.notna(base_val) and pd.notna(curr_val):
                drow[f"delta_{col}"] = curr_val - base_val
            else:
                drow[f"delta_{col}"] = float("nan")
        delta_rows.append(drow)

    delta_df = pd.DataFrame(delta_rows)
    return delta_df


# --- Copied-label analysis ---


def extract_noisy_labels(noisy_labels_dir):
    """Extract repeated noisy labels from noisy_only.tsv files."""
    configs = ["uu", "ur", "rr", "ru"]
    noisy_labels = {}

    for config in configs:
        tsv_path = os.path.join(noisy_labels_dir, config, "noisy_only.tsv")
        if not os.path.exists(tsv_path):
            print(f"  WARNING: {tsv_path} not found, skipping {config}", flush=True)
            continue

        labels = set()
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence = row["sentence"].strip().lower()
                labels.add(sentence)

        noisy_labels[config] = list(labels)
        print(f"  {config}: {len(labels)} unique noisy labels", flush=True)

    return noisy_labels


def compute_copied_label_rates(df, noisy_labels, output_dir):
    """Check if hallucination-like UR outputs match training noisy labels."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    results = []

    for cond in df["noise_condition"].unique():
        cond_lower = cond.lower()
        if cond_lower not in noisy_labels:
            print(f"  No noisy labels for {cond}, skipping copied-label analysis", flush=True)
            continue

        labels = noisy_labels[cond_lower]
        if not labels:
            continue

        # Get hallucination-like outputs for this condition
        hall_mask = df["noise_condition"] == cond
        if "hallucination_like" in df.columns:
            hall_mask = hall_mask & df["hallucination_like"]

        hall_df = df[hall_mask]
        if len(hall_df) == 0:
            print(f"  No hallucination-like outputs for {cond}", flush=True)
            results.append({
                "condition": cond,
                "n_hallucination_like": 0,
                "n_copied": 0,
                "copied_label_rate": 0.0,
                "mean_max_similarity": 0.0,
            })
            continue

        # TF-IDF vectorize labels and hypotheses
        hyp_texts = [normalize_text(h) for h in hall_df["hypothesis"].tolist()]
        label_texts = [normalize_text(l) for l in labels]

        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        all_texts = label_texts + hyp_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        label_vectors = tfidf_matrix[:len(label_texts)]
        hyp_vectors = tfidf_matrix[len(label_texts):]

        # Compute max similarity for each hypothesis against all labels
        similarities = cosine_similarity(hyp_vectors, label_vectors)
        max_sims = similarities.max(axis=1)

        threshold = 0.7
        n_copied = (max_sims >= threshold).sum()
        copied_rate = n_copied / len(hall_df) if len(hall_df) > 0 else 0.0

        print(f"  {cond}: {n_copied}/{len(hall_df)} copied-label ({copied_rate:.3f})", flush=True)

        # Save per-hypothesis similarities
        copied_path = os.path.join(output_dir, f"copied_label_detail_{cond}.csv")
        with open(copied_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hypothesis", "max_similarity", "nearest_label_idx", "nearest_label"])
            for i in range(len(hall_df)):
                best_idx = similarities[i].argmax()
                writer.writerow([
                    hall_df.iloc[i]["hypothesis"],
                    max_sims[i],
                    best_idx,
                    labels[best_idx],
                ])
        print(f"  Saved details to {copied_path}", flush=True)

        results.append({
            "condition": cond,
            "n_hallucination_like": len(hall_df),
            "n_copied": int(n_copied),
            "copied_label_rate": copied_rate,
            "mean_max_similarity": float(max_sims.mean()),
        })

    return pd.DataFrame(results)


# --- Qualitative examples ---


def extract_qualitative_examples(df, primary_norm_col, output_dir):
    """Extract strongest examples for each failure mode."""
    examples = []

    # UR hallucination-like (lowest WAcc + highest fluency)
    ur_hall = df[(df["noise_condition"] == "UR") & (df.get("hallucination_like", False))]
    if len(ur_hall) == 0:
        ur_hall = df[df["noise_condition"] == "UR"]  # fallback if no hallucination flag

    if len(ur_hall) > 0:
        # Rank by (1 - WAcc) * fluency composite
        if primary_norm_col in ur_hall.columns:
            ur_hall = ur_hall.copy()
            ur_hall["hall_score"] = (1 - ur_hall["wacc"]) * ur_hall[primary_norm_col]
            ur_top5 = ur_hall.nlargest(5, "hall_score")
        else:
            ur_top5 = ur_hall.nsmallest(5, "wacc")  # fallback: lowest WAcc

        for _, row in ur_top5.iterrows():
            examples.append({
                "failure_mode": "UR_hallucination_like",
                "utt_id": row["utt_id"],
                "reference": row["reference"],
                "hypothesis": row["hypothesis"],
                "wacc": row["wacc"],
                "normalized_sentence_score": row.get(primary_norm_col, float("nan")),
                "cosine_similarity": row["cosine_similarity"],
                "trigram_rep_count": row["trigram_rep_count"],
                "fourgram_rep_count": row["fourgram_rep_count"],
            })

    # RR repetition (most trigram/four-gram reps)
    rr_df = df[df["noise_condition"] == "RR"]
    if len(rr_df) > 0:
        rr_df = rr_df.copy()
        rr_df["total_rep"] = rr_df["trigram_rep_count"] + rr_df["fourgram_rep_count"]
        rr_top5 = rr_df.nlargest(5, "total_rep")
        for _, row in rr_top5.iterrows():
            examples.append({
                "failure_mode": "RR_repetition",
                "utt_id": row["utt_id"],
                "reference": row["reference"],
                "hypothesis": row["hypothesis"],
                "wacc": row["wacc"],
                "normalized_sentence_score": row.get(primary_norm_col, float("nan")),
                "cosine_similarity": row["cosine_similarity"],
                "trigram_rep_count": row["trigram_rep_count"],
                "fourgram_rep_count": row["fourgram_rep_count"],
            })

    # Baseline normal errors (representative errors, not extremes)
    base_df = df[df["noise_condition"] == "base"]
    if len(base_df) > 0:
        # Pick median-WER examples for representativeness
        base_sorted = base_df.sort_values("wacc")
        n = len(base_sorted)
        indices = [int(n * 0.05), int(n * 0.25), int(n * 0.5), int(n * 0.75), int(n * 0.95)]
        for idx in indices:
            idx = min(idx, n - 1)
            row = base_sorted.iloc[idx]
            examples.append({
                "failure_mode": "baseline_normal_errors",
                "utt_id": row["utt_id"],
                "reference": row["reference"],
                "hypothesis": row["hypothesis"],
                "wacc": row["wacc"],
                "normalized_sentence_score": row.get(primary_norm_col, float("nan")),
                "cosine_similarity": row["cosine_similarity"],
                "trigram_rep_count": row["trigram_rep_count"],
                "fourgram_rep_count": row["fourgram_rep_count"],
            })

    examples_df = pd.DataFrame(examples)
    examples_path = os.path.join(output_dir, "qualitative_examples.csv")
    examples_df.to_csv(examples_path, index=False)
    print(f"Saved {len(examples_df)} qualitative examples to {examples_path}", flush=True)
    return examples_df


# --- Cross-model scaffold ---


def build_cross_model_scaffold(agg_df, output_dir):
    """Build cross-model comparison CSV with Whisper data populated, fairseq as placeholder."""
    rows = []
    for _, row in agg_df.iterrows():
        cond = row["condition"]
        rows.append({
            "model": "Whisper",
            "condition": cond,
            "mean_wacc": row.get("mean_wacc", float("nan")),
            "mean_normalized_score": row.get("mean_normalized_score_gpt2", float("nan")),
            "hallucination_like_rate": row.get("hallucination_like_rate", float("nan")),
            "trigram_rep_rate": row.get("trigram_rep_rate", float("nan")),
            "fourgram_rep_rate": row.get("fourgram_rep_rate", float("nan")),
            "mean_cosine_similarity": row.get("mean_cosine_similarity", float("nan")),
        })

    # Add placeholder fairseq rows
    for cond in ["base", "UU", "UR", "RR", "RU"]:
        rows.append({
            "model": "Fairseq",
            "condition": cond,
            "mean_wacc": "TBD",
            "mean_normalized_score": "TBD",
            "hallucination_like_rate": "TBD",
            "trigram_rep_rate": "TBD",
            "fourgram_rep_rate": "TBD",
            "mean_cosine_similarity": "TBD",
        })

    cross_df = pd.DataFrame(rows)
    cross_path = os.path.join(output_dir, "cross_model_comparison.csv")
    cross_df.to_csv(cross_path, index=False)
    print(f"Saved cross-model comparison scaffold to {cross_path}", flush=True)


# --- Fluency scorer robustness ---


def build_fluency_robustness(df, agg_df, weak_lms, strong_lms, output_dir):
    """Build table showing hallucination rates for weak vs strong LMs."""
    rows = []
    all_lms = weak_lms + strong_lms

    for cond in sorted(df["model_name"].unique()):
        cond_df = df[df["model_name"] == cond]
        noise_conditions = sorted(cond_df["noise_condition"].dropna().unique())
        noise_ratios = sorted(cond_df["noise_ratio"].dropna().unique())
        row = {
            "condition": cond,
            "model_name": cond,
            "noise_condition": noise_conditions[0] if len(noise_conditions) == 1 else ";".join(map(str, noise_conditions)),
            "noise_ratio": noise_ratios[0] if len(noise_ratios) == 1 else ";".join(map(str, noise_ratios)),
            "n_samples": len(cond_df),
        }

        # Get baseline thresholds for each LM
        base_df = df[df["noise_condition"] == "base"]
        for lm in all_lms:
            norm_col = f"normalized_sentence_score_{lm}"
            if norm_col not in df.columns:
                continue

            # Thresholds from baseline
            if len(base_df) > 0:
                wacc_thresh = base_df["wacc"].mean()
                flu_thresh = base_df[norm_col].mean()
            else:
                wacc_thresh = df["wacc"].mean()
                flu_thresh = df[norm_col].mean()

            # Hall rate for this LM
            hall_mask = (
                (cond_df["wacc"] < wacc_thresh)
                & (cond_df[norm_col] > flu_thresh)
            )
            hall_rate = hall_mask.mean()

            row[f"mean_fluency_{lm}"] = cond_df[norm_col].mean()
            row[f"hall_rate_{lm}"] = hall_rate

        rows.append(row)

    robust_df = pd.DataFrame(rows)
    robust_path = os.path.join(output_dir, "fluency_scorer_robustness.csv")
    robust_df.to_csv(robust_path, index=False)
    print(f"Saved fluency robustness to {robust_path}", flush=True)
    return robust_df


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze cross-model validation metrics"
    )
    parser.add_argument("--per_utterance_dir", type=str, required=True,
                        help="Directory with per_utterance_*.csv files")
    parser.add_argument("--noisy_labels_dir", type=str, default=None,
                        help="Root directory with noisy_only.tsv files (for copied-label analysis)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for aggregate CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load ---
    print("Loading per-utterance data...", flush=True)
    df = load_all_per_utterance(args.per_utterance_dir)

    # --- Aggregate ---
    print("\nComputing aggregate metrics...", flush=True)
    agg_df, weak_lms, strong_lms, primary_lm, primary_norm_col = compute_aggregate_metrics(df)

    # --- Hallucination rates ---
    print("\nComputing hallucination-like rates...", flush=True)
    df, agg_df, wacc_thresh, flu_thresh = compute_hallucination_rates(
        df, agg_df, primary_lm, primary_norm_col
    )

    # --- Save aggregate ---
    agg_path = os.path.join(args.output_dir, "aggregate_metrics_whisper.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregate metrics to {agg_path}", flush=True)

    # --- Baseline deltas ---
    print("\nComputing baseline relative deltas...", flush=True)
    delta_df = compute_baseline_deltas(agg_df)
    delta_path = os.path.join(args.output_dir, "baseline_relative_deltas_whisper.csv")
    delta_df.to_csv(delta_path, index=False)
    print(f"Saved deltas to {delta_path}", flush=True)

    # --- Fluency robustness ---
    print("\nBuilding fluency scorer robustness...", flush=True)
    robust_df = build_fluency_robustness(df, agg_df, weak_lms, strong_lms, args.output_dir)

    # --- Copied-label ---
    if args.noisy_labels_dir and os.path.isdir(args.noisy_labels_dir):
        print("\nExtracting noisy labels for copied-label analysis...", flush=True)
        noisy_labels = extract_noisy_labels(args.noisy_labels_dir)
        if noisy_labels:
            print("Computing copied-label rates...", flush=True)
            copied_df = compute_copied_label_rates(df, noisy_labels, args.output_dir)
            copied_path = os.path.join(args.output_dir, "copied_label_analysis.csv")
            copied_df.to_csv(copied_path, index=False)
            print(f"Saved copied-label analysis to {copied_path}", flush=True)
    else:
        print("\nSkipping copied-label analysis (no --noisy_labels_dir provided)", flush=True)

    # --- Qualitative examples ---
    print("\nExtracting qualitative examples...", flush=True)
    examples_df = extract_qualitative_examples(df, primary_norm_col, args.output_dir)

    # --- Cross-model scaffold ---
    print("\nBuilding cross-model comparison scaffold...", flush=True)
    build_cross_model_scaffold(agg_df, args.output_dir)

    # --- Save enriched per-utterance ---
    enriched_path = os.path.join(args.output_dir, "per_utterance_metrics_whisper.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\nSaved enriched per-utterance data to {enriched_path}", flush=True)

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY: Aggregate Metrics")
    print(f"{'=' * 70}")
    display_cols = ["condition", "n_samples", "mean_wacc", "mean_cosine_similarity"]
    if f"mean_normalized_score_{primary_lm}" in agg_df.columns:
        display_cols.append(f"mean_normalized_score_{primary_lm}")
    if "hallucination_like_rate" in agg_df.columns:
        display_cols.append("hallucination_like_rate")
    display_cols.extend(["trigram_rep_rate", "fourgram_rep_rate"])

    available_cols = [c for c in display_cols if c in agg_df.columns]
    print(agg_df[available_cols].to_string(index=False))

    print(f"\nOutputs saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
