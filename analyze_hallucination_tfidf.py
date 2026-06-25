"""
Analyze where hallucinated Whisper transcripts may come from using TF-IDF.

The script compares hypotheses from evaluation detail TSVs against one or more
training transcript corpora. It writes nearest-neighbor matches and aggregate
summary tables that help separate likely copied/near-copied training text from
novel or generic hallucinations.

Example:
    python analyze_hallucination_tfidf.py \
        --details_glob '/scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_64pct/details_*full_noise*.tsv' \
        --train_corpus clean=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/base/train.tsv \
        --train_corpus rr_64pct=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_64pct/rr/train.tsv \
        --train_corpus ru_64pct=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_64pct/ru/train.tsv \
        --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/tfidf_provenance_64pct
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - exercised by runtime env only
    raise SystemExit(
        "Missing scikit-learn. Activate the project environment first, e.g.\n"
        "  source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh && conda activate llama\n"
        "or install dependencies from requirements.txt."
    ) from exc


PERTURBATION_PREFIXES = [
    "none",
    "full_noise_amp",
    "onset_noise_amp",
    "reverb_amp",
    "silence_amp",
    "leading_silence_amp",
    "speech_band_noise_amp",
]


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set:
    return set(text.split()) if text else set()


def token_overlap_scores(hypothesis: str, train_text: str) -> Tuple[float, float]:
    hyp_tokens = token_set(hypothesis)
    train_tokens = token_set(train_text)
    if not hyp_tokens or not train_tokens:
        return 0.0, 0.0
    intersection = hyp_tokens & train_tokens
    union = hyp_tokens | train_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    hyp_coverage = len(intersection) / len(hyp_tokens) if hyp_tokens else 0.0
    return jaccard, hyp_coverage


def classify_similarity(similarity: float) -> str:
    if similarity >= 0.98:
        return "exact_or_template_copy"
    if similarity >= 0.85:
        return "near_copy"
    if similarity >= 0.65:
        return "strong_overlap"
    if similarity >= 0.45:
        return "weak_overlap"
    return "novel_or_generic"


def parse_labeled_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = value
        label = Path(path).parent.name or Path(path).stem
        return label, path
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Expected LABEL=PATH for --train_corpus, got: {value}")
    return label, path


def read_tsv_sentences(path: str, text_column: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Column {text_column!r} not found in {path}; columns={reader.fieldnames}")

        for row_idx, row in enumerate(reader):
            raw_text = row.get(text_column, "")
            normalized = normalize_text(raw_text)
            if not normalized:
                continue
            rows.append(
                {
                    "train_row": row_idx,
                    "train_path": row.get("path", ""),
                    "train_sentence_id": row.get("sentence_id", ""),
                    "train_text": raw_text,
                    "train_text_norm": normalized,
                }
            )

    if not rows:
        raise ValueError(f"No non-empty {text_column!r} rows found in {path}")
    return pd.DataFrame(rows)


def detail_metadata(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if not stem.startswith("details_"):
        return "unknown", stem

    remainder = stem.removeprefix("details_")
    for prefix in PERTURBATION_PREFIXES:
        marker = f"_{prefix}"
        marker_index = remainder.find(marker)
        if marker_index > 0:
            return remainder[:marker_index], remainder[marker_index + 1:]

    return "unknown", remainder


def load_detail_rows(paths: Sequence[str], hallucination_filter: str, max_rows: int | None) -> pd.DataFrame:
    frames = []
    for value in paths:
        path = Path(value)
        config, perturbation = detail_metadata(path)
        df = pd.read_csv(path, sep="\t")
        required = {"reference", "hypothesis"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        df = df.copy()
        df["detail_file"] = str(path)
        df["eval_config"] = config
        df["perturbation"] = perturbation
        df["detail_row"] = np.arange(len(df))
        df["reference_norm"] = df["reference"].map(normalize_text)
        df["hypothesis_norm"] = df["hypothesis"].map(normalize_text)
        frames.append(df)

    if not frames:
        raise ValueError("No detail TSV files matched")

    details = pd.concat(frames, ignore_index=True)
    details = details[details["hypothesis_norm"].str.len() > 0].copy()

    if hallucination_filter == "hall_like":
        if "hallucination_like" not in details.columns:
            raise ValueError("hallucination_filter=hall_like requires a hallucination_like column")
        details = details[details["hallucination_like"].astype(float) > 0].copy()
    elif hallucination_filter == "wrong_or_repeated":
        repeat_cols = [col for col in ["2gram_reps", "3gram_reps", "4gram_reps"] if col in details.columns]
        repeated = details[repeat_cols].sum(axis=1) > 0 if repeat_cols else pd.Series(False, index=details.index)
        wrong = details["wacc"].astype(float) < 0 if "wacc" in details.columns else pd.Series(False, index=details.index)
        hall_like = (
            details["hallucination_like"].astype(float) > 0
            if "hallucination_like" in details.columns
            else pd.Series(False, index=details.index)
        )
        details = details[wrong | repeated | hall_like].copy()
    elif hallucination_filter == "all":
        pass
    else:
        raise ValueError(f"Unknown hallucination filter: {hallucination_filter}")

    if max_rows is not None and len(details) > max_rows:
        details = details.sort_values(["eval_config", "perturbation", "detail_row"]).head(max_rows).copy()

    details = details.reset_index(drop=True)
    details["analysis_id"] = np.arange(len(details))
    return details


def collect_detail_paths(details_glob: Sequence[str], details: Sequence[str]) -> List[str]:
    paths = []
    for pattern in details_glob:
        paths.extend(glob.glob(pattern))
    paths.extend(details)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No detail TSV files found from --details_glob/--details")
    return paths


def fit_corpus(label: str, path: str, text_column: str, ngram_range: Tuple[int, int], max_features: int | None):
    train_df = read_tsv_sentences(path, text_column=text_column)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(train_df["train_text_norm"].tolist())
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    return {
        "label": label,
        "path": path,
        "train_df": train_df,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "nn": nn,
    }


def nearest_matches(details: pd.DataFrame, corpus: Dict[str, object], top_k: int) -> pd.DataFrame:
    vectorizer = corpus["vectorizer"]
    nn = NearestNeighbors(n_neighbors=min(top_k, corpus["matrix"].shape[0]), metric="cosine", algorithm="brute")
    nn.fit(corpus["matrix"])

    query = vectorizer.transform(details["hypothesis_norm"].tolist())
    distances, indices = nn.kneighbors(query, return_distance=True)
    train_df = corpus["train_df"]

    rows = []
    for query_idx, analysis_id in enumerate(details["analysis_id"].tolist()):
        source = details.iloc[query_idx]
        for rank, (distance, train_idx) in enumerate(zip(distances[query_idx], indices[query_idx]), start=1):
            train_row = train_df.iloc[int(train_idx)]
            similarity = max(0.0, 1.0 - float(distance))
            jaccard, hyp_coverage = token_overlap_scores(source["hypothesis_norm"], train_row["train_text_norm"])
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "rank": rank,
                    "train_corpus": corpus["label"],
                    "train_source": corpus["path"],
                    "tfidf_similarity": similarity,
                    "similarity_class": classify_similarity(similarity),
                    "token_jaccard": jaccard,
                    "hypothesis_token_coverage": hyp_coverage,
                    "train_row": train_row["train_row"],
                    "train_path": train_row["train_path"],
                    "train_sentence_id": train_row["train_sentence_id"],
                    "train_text": train_row["train_text"],
                }
            )

    return pd.DataFrame(rows)


def summarize_best(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = best_df.groupby(["eval_config", "perturbation", "best_train_corpus"], dropna=False)
    for group_key, group in grouped:
        config, perturbation, corpus = group_key
        counts = group["best_similarity_class"].value_counts().to_dict()
        rows.append(
            {
                "eval_config": config,
                "perturbation": perturbation,
                "best_train_corpus": corpus,
                "n_hypotheses": len(group),
                "mean_best_tfidf_similarity": group["best_tfidf_similarity"].mean(),
                "median_best_tfidf_similarity": group["best_tfidf_similarity"].median(),
                "exact_or_template_copy": counts.get("exact_or_template_copy", 0),
                "near_copy": counts.get("near_copy", 0),
                "strong_overlap": counts.get("strong_overlap", 0),
                "weak_overlap": counts.get("weak_overlap", 0),
                "novel_or_generic": counts.get("novel_or_generic", 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["perturbation", "eval_config", "best_train_corpus"])


def summarize_topk_copy(matches_out: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    copy_classes = {"exact_or_template_copy", "near_copy"}
    total = matches_out[["analysis_id", "eval_config", "perturbation"]].drop_duplicates()

    copied = matches_out[matches_out["similarity_class"].isin(copy_classes)].copy()
    copied_any = copied[["analysis_id", "eval_config", "perturbation"]].drop_duplicates()

    overall_rows = []
    total_grouped = total.groupby(["eval_config", "perturbation"], dropna=False)
    copied_counts = copied_any.groupby(["eval_config", "perturbation"], dropna=False).size().to_dict()
    for group_key, group in total_grouped:
        eval_config, perturbation = group_key
        n_total = len(group)
        n_copy = int(copied_counts.get(group_key, 0))
        overall_rows.append(
            {
                "eval_config": eval_config,
                "perturbation": perturbation,
                "total_hypotheses": n_total,
                "top5_copy_count": n_copy,
                "top5_copy_ratio": n_copy / n_total if n_total else 0.0,
            }
        )

    by_source_rows = []
    total_by_config = total.groupby(["eval_config", "perturbation"], dropna=False).size().to_dict()
    copied_by_source = copied[["analysis_id", "eval_config", "perturbation", "train_corpus"]].drop_duplicates()
    for group_key, group in copied_by_source.groupby(["eval_config", "perturbation", "train_corpus"], dropna=False):
        eval_config, perturbation, train_corpus = group_key
        n_total = int(total_by_config[(eval_config, perturbation)])
        n_copy = len(group)
        by_source_rows.append(
            {
                "eval_config": eval_config,
                "perturbation": perturbation,
                "train_corpus": train_corpus,
                "total_hypotheses": n_total,
                "top5_copy_count": n_copy,
                "top5_copy_ratio": n_copy / n_total if n_total else 0.0,
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values(["perturbation", "eval_config"])
    by_source = pd.DataFrame(by_source_rows)
    if not by_source.empty:
        by_source = by_source.sort_values(["perturbation", "eval_config", "top5_copy_ratio"], ascending=[True, True, False])
    return overall, by_source


def summarize_attractors(best_df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    counts = Counter(best_df["hypothesis_norm"].tolist())
    rows = []
    for hypothesis_norm, count in counts.most_common():
        if count < min_count:
            continue
        sub = best_df[best_df["hypothesis_norm"] == hypothesis_norm].sort_values("best_tfidf_similarity", ascending=False)
        row = sub.iloc[0]
        rows.append(
            {
                "hypothesis_norm": hypothesis_norm,
                "count": count,
                "example_hypothesis": row["hypothesis"],
                "example_eval_config": row["eval_config"],
                "example_perturbation": row["perturbation"],
                "best_train_corpus": row["best_train_corpus"],
                "best_tfidf_similarity": row["best_tfidf_similarity"],
                "best_similarity_class": row["best_similarity_class"],
                "best_train_text": row["best_train_text"],
            }
        )
    return pd.DataFrame(rows)


def summarize_repeated_copy_attractors(best_df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    attractors = summarize_attractors(best_df, min_count=min_count)
    if attractors.empty:
        return attractors
    copy_classes = {"exact_or_template_copy", "near_copy"}
    return attractors[attractors["best_similarity_class"].isin(copy_classes)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF provenance analysis for Whisper hallucination outputs")
    parser.add_argument("--details_glob", action="append", default=[], help="Glob for detail TSV files. Can be repeated.")
    parser.add_argument("--details", action="append", default=[], help="Specific detail TSV file. Can be repeated.")
    parser.add_argument("--train_corpus", action="append", required=True, help="Training corpus as LABEL=PATH. Can be repeated.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_column", default="sentence")
    parser.add_argument("--hallucination_filter", choices=["hall_like", "wrong_or_repeated", "all"], default="wrong_or_repeated")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=3)
    parser.add_argument("--max_features", type=int, default=250000)
    parser.add_argument("--max_detail_rows", type=int, default=None)
    parser.add_argument("--min_attractor_count", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detail_paths = collect_detail_paths(args.details_glob, args.details)
    details = load_detail_rows(detail_paths, args.hallucination_filter, args.max_detail_rows)
    print(f"Loaded {len(details)} candidate hypotheses from {len(detail_paths)} detail files", flush=True)

    detail_cols = [
        "analysis_id", "detail_file", "detail_row", "eval_config", "perturbation",
        "reference", "hypothesis", "wer", "wacc", "norm_plausibility",
        "hallucination_like", "2gram_reps", "3gram_reps", "4gram_reps",
        "reference_norm", "hypothesis_norm",
    ]
    available_detail_cols = [col for col in detail_cols if col in details.columns]
    details[available_detail_cols].to_csv(Path(args.output_dir) / "candidate_hypotheses.tsv", sep="\t", index=False)

    corpora = []
    for value in args.train_corpus:
        label, path = parse_labeled_path(value)
        print(f"Indexing {label}: {path}", flush=True)
        corpora.append(
            fit_corpus(
                label=label,
                path=path,
                text_column=args.text_column,
                ngram_range=(args.ngram_min, args.ngram_max),
                max_features=args.max_features,
            )
        )

    all_matches = []
    for corpus in corpora:
        print(f"Searching nearest training transcripts in {corpus['label']}", flush=True)
        all_matches.append(nearest_matches(details, corpus, top_k=args.top_k))

    matches = pd.concat(all_matches, ignore_index=True)
    matches_out = details.merge(matches, on="analysis_id", how="inner")
    matches_out.to_csv(Path(args.output_dir) / "tfidf_matches_long.tsv", sep="\t", index=False)

    rank1 = matches_out[matches_out["rank"] == 1].copy()
    best_idx = rank1.groupby("analysis_id")["tfidf_similarity"].idxmax()
    best = rank1.loc[best_idx].copy()
    best = best.rename(
        columns={
            "train_corpus": "best_train_corpus",
            "train_source": "best_train_source",
            "tfidf_similarity": "best_tfidf_similarity",
            "similarity_class": "best_similarity_class",
            "token_jaccard": "best_token_jaccard",
            "hypothesis_token_coverage": "best_hypothesis_token_coverage",
            "train_row": "best_train_row",
            "train_path": "best_train_path",
            "train_sentence_id": "best_train_sentence_id",
            "train_text": "best_train_text",
        }
    )
    best.to_csv(Path(args.output_dir) / "tfidf_best_overall.tsv", sep="\t", index=False)

    summary = summarize_best(best)
    summary.to_csv(Path(args.output_dir) / "tfidf_provenance_summary.tsv", sep="\t", index=False)

    top5_copy, top5_copy_by_source = summarize_topk_copy(matches_out)
    top5_copy.to_csv(Path(args.output_dir) / "tfidf_top5_copy_summary.tsv", sep="\t", index=False)
    top5_copy_by_source.to_csv(Path(args.output_dir) / "tfidf_top5_copy_by_source.tsv", sep="\t", index=False)

    attractors = summarize_attractors(best, min_count=args.min_attractor_count)
    attractors.to_csv(Path(args.output_dir) / "tfidf_attractor_phrases.tsv", sep="\t", index=False)

    repeated_copy_attractors = summarize_repeated_copy_attractors(best, min_count=args.min_attractor_count)
    repeated_copy_attractors.to_csv(Path(args.output_dir) / "tfidf_repeated_copy_attractors.tsv", sep="\t", index=False)

    print(f"Saved outputs to {args.output_dir}", flush=True)
    print(f"  candidate_hypotheses.tsv: {len(details)} rows", flush=True)
    print(f"  tfidf_matches_long.tsv: {len(matches_out)} rows", flush=True)
    print(f"  tfidf_best_overall.tsv: {len(best)} rows", flush=True)
    print(f"  tfidf_provenance_summary.tsv: {len(summary)} rows", flush=True)
    print(f"  tfidf_top5_copy_summary.tsv: {len(top5_copy)} rows", flush=True)
    print(f"  tfidf_top5_copy_by_source.tsv: {len(top5_copy_by_source)} rows", flush=True)
    print(f"  tfidf_attractor_phrases.tsv: {len(attractors)} rows", flush=True)
    print(f"  tfidf_repeated_copy_attractors.tsv: {len(repeated_copy_attractors)} rows", flush=True)


if __name__ == "__main__":
    main()