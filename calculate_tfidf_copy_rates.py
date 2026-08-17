#!/usr/bin/env python3
"""Compute TF-IDF training-phrase copy rates for hallucination-like outputs.

For each table model, this script filters to hallucination-like hypotheses and
compares those hypotheses against the corresponding training TSV for that model
family. It reports exact/template copies (TF-IDF cosine >= 0.98) and
near-copies including exact/template copies (TF-IDF cosine >= 0.85).
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
DATA_ROOT = Path("/scratch/vemotionsys/rmfrieske/datasets")
WORKSPACE = Path("/home/rmfrieske/whisper_hallucination")

WHISPER_BASE_FILE = SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"
WHISPER_EVAL_64PCT = SCRATCH_ROOT / "eval_64pct"
FAIRSEQ_PER_UTTERANCE = WORKSPACE / "fairseq_eval_lm" / "per_utterance_metrics_fairseq.csv"

OUTPUT_TSV = WORKSPACE / "plots" / "tfidf_copy_rates_key_table.tsv"
MATCHES_TSV = WORKSPACE / "plots" / "tfidf_copy_matches_key_table.tsv"

WHISPER_FILES = {
    "Base": [WHISPER_BASE_FILE],
    "RR": [WHISPER_EVAL_64PCT / "per_utterance_rr_64pct_checkpoint-9375.csv"],
    "RU": [WHISPER_EVAL_64PCT / "per_utterance_ru_64pct_checkpoint-9375.csv"],
    "UR": [
        WHISPER_EVAL_64PCT / "per_utterance_ur_64pct_checkpoint-10000_shard00-of-02.csv",
        WHISPER_EVAL_64PCT / "per_utterance_ur_64pct_checkpoint-10000_shard01-of-02.csv",
    ],
    "UU": [
        WHISPER_EVAL_64PCT / "per_utterance_uu_64pct_final_shard00-of-02.csv",
        WHISPER_EVAL_64PCT / "per_utterance_uu_64pct_final_shard01-of-02.csv",
    ],
}

WHISPER_TRAIN = {
    "Base": DATA_ROOT / "whisper_hallucination" / "base" / "train.tsv",
    "RR": DATA_ROOT / "whisper_hallucination_64pct" / "rr" / "train.tsv",
    "RU": DATA_ROOT / "whisper_hallucination_64pct" / "ru" / "train.tsv",
    "UR": DATA_ROOT / "whisper_hallucination_64pct" / "ur" / "train.tsv",
    "UU": DATA_ROOT / "whisper_hallucination_64pct" / "uu" / "train.tsv",
}

FAIRSEQ_TRAIN = {
    "Base": DATA_ROOT / "whisper_hallucination" / "base" / "train.tsv",
    "RR": DATA_ROOT / "whisper_hallucination" / "rr" / "train.tsv",
    "RU": DATA_ROOT / "whisper_hallucination" / "ru" / "train.tsv",
    "UR": DATA_ROOT / "whisper_hallucination" / "ur" / "train.tsv",
    "UU": DATA_ROOT / "whisper_hallucination" / "uu" / "train.tsv",
}

FAIRSEQ_CONDITION_MAP = {
    "Base": "base",
    "RR": "rr",
    "RU": "ru",
    "UR": "ur",
    "UU": "uu",
}


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_training_texts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    if "sentence" not in df.columns:
        raise ValueError(f"Missing sentence column in {path}")
    out = df[["sentence"]].copy()
    out["train_text_norm"] = out["sentence"].map(normalize_text)
    out = out[out["train_text_norm"].str.len() > 0].reset_index(drop=True)
    out["train_row"] = out.index
    return out


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def add_whisper_hallucination_flags(df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    wer_threshold = base_df["wer"].mean()
    fluency_threshold = base_df["normalized_sentence_score_Qwen3-0.6B"].mean()
    flagged = df.copy()
    flagged["hallucination_like"] = (
        (flagged["wer"] > wer_threshold)
        & (flagged["normalized_sentence_score_Qwen3-0.6B"] > fluency_threshold)
    )
    return flagged


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def best_tfidf_matches(hypotheses: pd.Series, train_df: pd.DataFrame, ngram_max: int, max_features: int) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, ngram_max),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_df["train_text_norm"].tolist())
    query = vectorizer.transform(hypotheses.tolist())
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(train_matrix)
    distances, indices = nn.kneighbors(query, return_distance=True)

    rows = []
    for row_idx, (distance, train_idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        train_row = train_df.iloc[int(train_idx)]
        rows.append({
            "candidate_index": row_idx,
            "best_tfidf_similarity": max(0.0, 1.0 - float(distance)),
            "best_train_row": int(train_row["train_row"]),
            "best_train_text": train_row["sentence"],
        })
    return pd.DataFrame(rows)


def summarize_model(model_family: str, condition: str, candidates: pd.DataFrame, train_path: Path, args) -> tuple[dict, pd.DataFrame]:
    train_df = read_training_texts(train_path)
    candidates = candidates.copy()
    candidates["hypothesis_norm_for_tfidf"] = candidates["hypothesis"].map(normalize_text)
    candidates = candidates[candidates["hypothesis_norm_for_tfidf"].str.len() > 0].reset_index(drop=True)

    if candidates.empty:
        return {
            "Model": model_family,
            "Condition": condition,
            "train_corpus": str(train_path),
            "hallucination_like_n": 0,
            "exact_or_template_copy_count": 0,
            "near_copy_including_exact_count": 0,
            "exact_or_template_copy_rate_pct": 0.0,
            "near_copy_including_exact_rate_pct": 0.0,
            "mean_best_tfidf_similarity": 0.0,
            "median_best_tfidf_similarity": 0.0,
        }, pd.DataFrame()

    matches = best_tfidf_matches(
        candidates["hypothesis_norm_for_tfidf"],
        train_df,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    merged = pd.concat([candidates.reset_index(drop=True), matches], axis=1)
    exact = merged["best_tfidf_similarity"] >= args.exact_threshold
    near_including_exact = merged["best_tfidf_similarity"] >= args.near_threshold

    summary = {
        "Model": model_family,
        "Condition": condition,
        "train_corpus": str(train_path),
        "hallucination_like_n": len(merged),
        "exact_or_template_copy_count": int(exact.sum()),
        "near_copy_including_exact_count": int(near_including_exact.sum()),
        "exact_or_template_copy_rate_pct": 100.0 * float(exact.mean()),
        "near_copy_including_exact_rate_pct": 100.0 * float(near_including_exact.mean()),
        "mean_best_tfidf_similarity": float(merged["best_tfidf_similarity"].mean()),
        "median_best_tfidf_similarity": float(merged["best_tfidf_similarity"].median()),
    }

    keep_cols = [
        "Model", "Condition", "hypothesis", "reference", "wacc",
        "normalized_sentence_score_Qwen3-0.6B", "best_tfidf_similarity",
        "best_train_text", "best_train_row",
    ]
    merged["Model"] = model_family
    merged["Condition"] = condition
    return summary, merged[[col for col in keep_cols if col in merged.columns]]


def whisper_candidates() -> dict[str, pd.DataFrame]:
    base_df = pd.read_csv(WHISPER_BASE_FILE)
    out = {}
    for condition, paths in WHISPER_FILES.items():
        df = load_csvs(paths)
        df = add_whisper_hallucination_flags(df, base_df)
        out[condition] = df[df["hallucination_like"]].copy()
    return out


def fairseq_candidates() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(FAIRSEQ_PER_UTTERANCE)
    df = df[bool_series(df["hallucination_like"])].copy()
    out = {}
    for condition, model_name in FAIRSEQ_CONDITION_MAP.items():
        out[condition] = df[df["model_name"] == model_name].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_tsv", type=Path, default=OUTPUT_TSV)
    parser.add_argument("--matches_tsv", type=Path, default=MATCHES_TSV)
    parser.add_argument("--ngram_max", type=int, default=3)
    parser.add_argument("--max_features", type=int, default=250000)
    parser.add_argument("--near_threshold", type=float, default=0.85)
    parser.add_argument("--exact_threshold", type=float, default=0.98)
    args = parser.parse_args()

    summaries = []
    match_frames = []

    for condition, candidates in whisper_candidates().items():
        summary, matches = summarize_model("Whisper 64%", condition, candidates, WHISPER_TRAIN[condition], args)
        summaries.append(summary)
        if not matches.empty:
            match_frames.append(matches)

    for condition, candidates in fairseq_candidates().items():
        summary, matches = summarize_model("fairseq 8%", condition, candidates, FAIRSEQ_TRAIN[condition], args)
        summaries.append(summary)
        if not matches.empty:
            match_frames.append(matches)

    summary_df = pd.DataFrame(summaries)
    model_order = {"Whisper 64%": 0, "fairseq 8%": 1}
    cond_order = {"Base": 0, "RR": 1, "RU": 2, "UR": 3, "UU": 4}
    summary_df["_model_order"] = summary_df["Model"].map(model_order)
    summary_df["_cond_order"] = summary_df["Condition"].map(cond_order)
    summary_df = summary_df.sort_values(["_model_order", "_cond_order"]).drop(columns=["_model_order", "_cond_order"])

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_tsv, sep="\t", index=False, float_format="%.6f")
    if match_frames:
        pd.concat(match_frames, ignore_index=True).to_csv(args.matches_tsv, sep="\t", index=False, float_format="%.6f")

    display_cols = [
        "Model", "Condition", "hallucination_like_n",
        "exact_or_template_copy_count", "exact_or_template_copy_rate_pct",
        "near_copy_including_exact_count", "near_copy_including_exact_rate_pct",
        "mean_best_tfidf_similarity", "median_best_tfidf_similarity",
    ]
    print(summary_df[display_cols].to_csv(sep="\t", index=False, float_format="%.2f"), end="")
    print(f"\nSaved: {args.output_tsv}")
    print(f"Saved: {args.matches_tsv}")


if __name__ == "__main__":
    main()