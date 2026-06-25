"""
Evaluate fairseq ASR TSV outputs with the same text metrics used for Whisper.

Reads fairseq_results/results_*.tsv files with columns:
  ref_sentence, hypo_sentence, probability_ref, probability_hypo,
  scaled_probability_hypo, cosine_similarity

Outputs:
  - per_utterance_metrics_fairseq.csv
  - aggregate_metrics_fairseq.csv
  - baseline_relative_deltas_fairseq.csv
  - summary_fairseq.json

Fast mode uses the fairseq-provided cosine_similarity and scaled_probability_hypo.
Use --recompute_cosine and/or --recompute_lm to regenerate those signals with the
same sentence-transformer and causal-LM scoring conventions as
evaluate_whisper_validation.py.
"""

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WHISPER_SPECIAL = re.compile(r"<\|[^|]+\|>")
DEFAULT_CONDITION_ORDER = ["base", "rr", "ru", "ur", "uu"]


def normalize_text(text: str) -> str:
    """Normalize text for WER/WAcc computation. Does not collapse repeated words."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = WHISPER_SPECIAL.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_metrics(hypotheses, references):
    """Compute per-utterance WER, WAcc, S, D, I counts."""
    import jiwer

    results = []
    for hyp, ref in zip(hypotheses, references):
        hyp_norm = normalize_text(hyp)
        ref_norm = normalize_text(ref)

        if len(ref_norm.split()) == 0:
            results.append({
                "wer": 1.0,
                "wacc": 0.0,
                "s_count": 0,
                "d_count": 0,
                "i_count": len(hyp_norm.split()),
                "num_ref_words": 0,
                "num_hyp_words": len(hyp_norm.split()),
            })
            continue

        measures = jiwer.compute_measures(ref_norm, hyp_norm)
        wer_val = measures["wer"]
        results.append({
            "wer": wer_val,
            "wacc": 1.0 - wer_val,
            "s_count": measures.get("substitutions", 0),
            "d_count": measures.get("deletions", 0),
            "i_count": measures.get("insertions", 0),
            "num_ref_words": len(ref_norm.split()),
            "num_hyp_words": len(hyp_norm.split()),
        })

    return results


def compute_bleu_scores(hypotheses, references):
    """Compute sentence-level BLEU-4 with smoothing via sacrebleu."""
    import sacrebleu

    results = []
    for hyp, ref in zip(hypotheses, references):
        hyp_norm = normalize_text(hyp)
        ref_norm = normalize_text(ref)
        if not hyp_norm.strip():
            results.append(0.0)
            continue
        bleu_val = sacrebleu.sentence_bleu(hyp_norm, [ref_norm], smooth_method="exp").score
        results.append(bleu_val / 100.0)
    return results


def compute_repetition_metrics(text):
    """
    Compute detailed n-gram repetition metrics.
    repetition_count = sum(count - 1 for each ngram where count > 1)
    has_repetition = repetition_count >= 2
    """
    tokens = normalize_text(text).split()
    results = {}

    for n in [2, 3, 4]:
        prefix = ["bigram", "trigram", "fourgram"][n - 2]
        if len(tokens) < n:
            results[f"{prefix}_rep_count"] = 0
            results[f"has_{prefix}_rep"] = False
            continue

        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        rep_count = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
        results[f"{prefix}_rep_count"] = rep_count
        results[f"has_{prefix}_rep"] = rep_count >= 2

    return results


def compute_cosine_similarities(hypotheses, references, model_name="all-MiniLM-L6-v2", device="cpu"):
    """Compute semantic cosine similarity using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name} ...", flush=True)
    start_time = time.time()
    embedder = SentenceTransformer(model_name, device=device)
    print(f"  Loaded in {time.time() - start_time:.0f}s", flush=True)

    norm_hyps = [normalize_text(h) for h in hypotheses]
    norm_refs = [normalize_text(r) for r in references]

    print(f"Encoding {len(norm_hyps)} hypotheses + references ...", flush=True)
    start_time = time.time()
    emb_hyps = embedder.encode(norm_hyps, show_progress_bar=True, batch_size=64)
    emb_refs = embedder.encode(norm_refs, show_progress_bar=True, batch_size=64)
    print(f"  Encoded in {time.time() - start_time:.0f}s", flush=True)

    similarities = []
    for hyp_emb, ref_emb in zip(emb_hyps, emb_refs):
        cos_sim = np.dot(hyp_emb, ref_emb) / (np.linalg.norm(hyp_emb) * np.linalg.norm(ref_emb) + 1e-8)
        similarities.append(float(cos_sim))

    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return similarities


def compute_lm_scores_cached(texts, model, tokenizer, batch_size=8):
    """
    Compute sentence probability scores using an already-loaded causal LM.
    Returns sentence_score = exp(-avg_token_nll) and avg token NLL.
    """
    scores = []
    nlls = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(batch_texts):
            text = text.strip()
            if text:
                valid_indices.append(idx)
                valid_texts.append(text)

        batch_scores = [1e-8] * len(batch_texts)
        batch_nlls = [20.0] * len(batch_texts)

        if valid_texts:
            encodings = tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            encodings = {key: value.to(model.device) for key, value in encodings.items()}

            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            logits = outputs.logits
            per_seq_nlls = []

            for i in range(len(valid_texts)):
                seq_len = (encodings["attention_mask"][i] == 1).sum().item()
                if seq_len <= 1:
                    per_seq_nlls.append(20.0)
                    continue
                shift_logits = logits[i, :seq_len - 1, :]
                shift_labels = encodings["input_ids"][i, 1:seq_len]
                per_seq_nlls.append(loss_fct(shift_logits, shift_labels).mean().item())

            for idx, nll_val in zip(valid_indices, per_seq_nlls):
                batch_scores[idx] = float(np.exp(-nll_val))
                batch_nlls[idx] = float(nll_val)

        scores.extend(batch_scores)
        nlls.extend(batch_nlls)

        if (batch_start // batch_size) % 10 == 0:
            print(f"    Scored {min(batch_start + batch_size, len(texts))}/{len(texts)}", flush=True)

    return scores, nlls


def score_lm_models(hypotheses, references, lm_models, device="cpu", batch_size=8):
    """Score hypotheses and references with each causal LM and return Whisper-style columns."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    norm_hyps = [normalize_text(h) for h in hypotheses]
    norm_refs = [normalize_text(r) for r in references]
    lm_scores = {}

    for lm_name in lm_models:
        lm_short = lm_name.split("/")[-1]
        print(f"\n=== LM Scoring: {lm_name} ===", flush=True)
        start_time = time.time()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(lm_name, trust_remote_code=True, dtype=dtype)
        model = model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Loaded in {time.time() - start_time:.0f}s", flush=True)

        print("  Scoring hypotheses...", flush=True)
        hyp_scores, hyp_nlls = compute_lm_scores_cached(norm_hyps, model, tokenizer, batch_size=batch_size)
        print("  Scoring references...", flush=True)
        ref_scores, ref_nlls = compute_lm_scores_cached(norm_refs, model, tokenizer, batch_size=batch_size)

        norm_scores = []
        for hyp_score, ref_score in zip(hyp_scores, ref_scores):
            norm_score = hyp_score / (ref_score + 1e-8)
            norm_scores.append(min(1.0, max(0.0, norm_score)))

        lm_scores[lm_short] = {
            "scores": hyp_scores,
            "nlls": hyp_nlls,
            "ref_scores": ref_scores,
            "ref_nlls": ref_nlls,
            "norm_scores": norm_scores,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Mean normalized score ({lm_short}): {np.mean(norm_scores):.4f}", flush=True)

    return lm_scores


def parse_condition(path):
    """Map results_RU.tsv-style names to model and noise metadata."""
    match = re.match(r"results_(.+)\.tsv$", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Input file must match results_*.tsv: {path}")

    raw_condition = match.group(1)
    model_name = raw_condition.lower()
    if model_name == "base":
        return model_name, "base"
    return model_name, raw_condition.upper()


def read_fairseq_tsv(path):
    df = pd.read_csv(path, sep="\t", keep_default_na=False, quoting=csv.QUOTE_NONE)
    required = {
        "ref_sentence",
        "hypo_sentence",
        "probability_ref",
        "probability_hypo",
        "scaled_probability_hypo",
        "cosine_similarity",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def add_fast_fluency_columns(out_df, source_col="scaled_probability_hypo"):
    values = pd.to_numeric(out_df[source_col], errors="coerce").fillna(0.0).astype(float)
    out_df["sentence_score_fairseq_scaled"] = values
    out_df["normalized_sentence_score_fairseq_scaled"] = values.clip(lower=0.0, upper=1.0)


def build_per_utterance(input_paths, args):
    rows = []

    for path in input_paths:
        model_name, noise_condition = parse_condition(path)
        noise_ratio = 0.0 if model_name == "base" else args.noise_ratio
        print(f"\nLoading {path.name} as model={model_name}, condition={noise_condition}", flush=True)

        df = read_fairseq_tsv(path)
        references = df["ref_sentence"].astype(str).tolist()
        hypotheses = df["hypo_sentence"].astype(str).tolist()

        print("  Computing WER/WAcc...", flush=True)
        wer_results = compute_wer_metrics(hypotheses, references)

        print("  Computing BLEU...", flush=True)
        bleu_scores = compute_bleu_scores(hypotheses, references)

        print("  Computing repetition metrics...", flush=True)
        rep_results = [compute_repetition_metrics(hyp) for hyp in hypotheses]

        if args.recompute_cosine:
            print("  Recomputing cosine similarities...", flush=True)
            cosine_sims = compute_cosine_similarities(
                hypotheses,
                references,
                model_name=args.embedding_model,
                device=args.device,
            )
        else:
            cosine_sims = pd.to_numeric(df["cosine_similarity"], errors="coerce").fillna(0.0).tolist()

        for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
            row = {
                "utt_id": f"{model_name}_{idx:06d}",
                "source_file": path.name,
                "reference": ref,
                "hypothesis": hyp,
                "model_name": model_name,
                "noise_condition": noise_condition,
                "noise_ratio": noise_ratio,
                "wer": wer_results[idx]["wer"],
                "wacc": wer_results[idx]["wacc"],
                "s_count": wer_results[idx]["s_count"],
                "d_count": wer_results[idx]["d_count"],
                "i_count": wer_results[idx]["i_count"],
                "num_ref_words": wer_results[idx]["num_ref_words"],
                "num_hyp_words": wer_results[idx]["num_hyp_words"],
                "cosine_similarity": cosine_sims[idx],
                "bleu": bleu_scores[idx],
                "bigram_rep_count": rep_results[idx]["bigram_rep_count"],
                "trigram_rep_count": rep_results[idx]["trigram_rep_count"],
                "fourgram_rep_count": rep_results[idx]["fourgram_rep_count"],
                "has_bigram_rep": rep_results[idx]["has_bigram_rep"],
                "has_trigram_rep": rep_results[idx]["has_trigram_rep"],
                "has_fourgram_rep": rep_results[idx]["has_fourgram_rep"],
                "decoded_text_raw": hyp,
                "decoded_text_normalized": normalize_text(hyp),
                "probability_ref": pd.to_numeric(df.iloc[idx]["probability_ref"], errors="coerce"),
                "probability_hypo": pd.to_numeric(df.iloc[idx]["probability_hypo"], errors="coerce"),
                "scaled_probability_hypo": pd.to_numeric(df.iloc[idx]["scaled_probability_hypo"], errors="coerce"),
            }
            rows.append(row)

    per_df = pd.DataFrame(rows)
    if args.recompute_lm:
        print("\nRecomputing LM fluency over all fairseq rows...", flush=True)
        lm_scores = score_lm_models(
            per_df["hypothesis"].astype(str).tolist(),
            per_df["reference"].astype(str).tolist(),
            lm_models=args.lm_models,
            device=args.device,
            batch_size=args.lm_batch_size,
        )
        for lm_short, lm_data in lm_scores.items():
            per_df[f"sentence_score_{lm_short}"] = lm_data["scores"]
            per_df[f"nll_{lm_short}"] = lm_data["nlls"]
            per_df[f"ref_sentence_score_{lm_short}"] = lm_data["ref_scores"]
            per_df[f"normalized_sentence_score_{lm_short}"] = lm_data["norm_scores"]
    else:
        add_fast_fluency_columns(per_df)
    return per_df


def choose_primary_lm(per_df, requested_primary=None):
    lm_cols = [col for col in per_df.columns if col.startswith("normalized_sentence_score_")]
    lm_names = [col.replace("normalized_sentence_score_", "") for col in lm_cols]
    if requested_primary:
        primary_col = f"normalized_sentence_score_{requested_primary}"
        if primary_col not in per_df.columns:
            raise ValueError(f"Requested primary LM column is missing: {primary_col}")
        return requested_primary, primary_col, lm_names

    strong_lms = [name for name in lm_names if "gpt2" not in name.lower() and "small" not in name.lower()]
    primary = strong_lms[0] if strong_lms else (lm_names[0] if lm_names else None)
    if primary is None:
        raise ValueError("No normalized fluency column found.")
    return primary, f"normalized_sentence_score_{primary}", lm_names


def compute_aggregate_metrics(per_df, primary_col):
    lm_cols = [col for col in per_df.columns if col.startswith("normalized_sentence_score_")]
    agg_rows = []
    ordered_models = [name for name in DEFAULT_CONDITION_ORDER if name in set(per_df["model_name"])]
    ordered_models.extend(sorted(set(per_df["model_name"]) - set(ordered_models)))

    for model_name in ordered_models:
        cond_df = per_df[per_df["model_name"] == model_name]
        row = {
            "condition": model_name,
            "model_name": model_name,
            "noise_condition": ";".join(map(str, sorted(cond_df["noise_condition"].dropna().unique()))),
            "noise_ratio": ";".join(map(str, sorted(cond_df["noise_ratio"].dropna().unique()))),
            "n_samples": len(cond_df),
            "mean_wer": cond_df["wer"].mean(),
            "mean_wacc": cond_df["wacc"].mean(),
            "mean_cosine_similarity": cond_df["cosine_similarity"].mean(),
            "mean_bleu": cond_df["bleu"].mean(),
            "mean_bigram_rep_count": cond_df["bigram_rep_count"].mean(),
            "mean_trigram_rep_count": cond_df["trigram_rep_count"].mean(),
            "mean_fourgram_rep_count": cond_df["fourgram_rep_count"].mean(),
            "bigram_rep_rate": cond_df["has_bigram_rep"].mean(),
            "trigram_rep_rate": cond_df["has_trigram_rep"].mean(),
            "fourgram_rep_rate": cond_df["has_fourgram_rep"].mean(),
        }
        for col in lm_cols:
            lm_name = col.replace("normalized_sentence_score_", "")
            row[f"mean_normalized_score_{lm_name}"] = cond_df[col].mean()
        row["primary_fluency_mean"] = cond_df[primary_col].mean()
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def compute_hallucination_rates(per_df, agg_df, primary_col):
    base_df = per_df[per_df["noise_condition"].str.lower() == "base"]
    if len(base_df) == 0:
        print("  WARNING: no base condition found; using global thresholds.", flush=True)
        base_df = per_df

    wacc_threshold = base_df["wacc"].mean()
    fluency_threshold = base_df[primary_col].mean()
    wacc_q25 = base_df["wacc"].quantile(0.25)
    fluency_median = base_df[primary_col].median()

    print(f"  WAcc threshold (mean base): {wacc_threshold:.4f}", flush=True)
    print(f"  Fluency threshold (mean base): {fluency_threshold:.4f}", flush=True)
    print(f"  WAcc Q25 (strict): {wacc_q25:.4f}", flush=True)
    print(f"  Fluency median (strict): {fluency_median:.4f}", flush=True)

    per_df["hallucination_like"] = False
    per_df["hallucination_like_strict"] = False

    for model_name in sorted(per_df["model_name"].unique()):
        cond_mask = per_df["model_name"] == model_name
        cond_df = per_df[cond_mask]
        hall_standard = (cond_df["wacc"] < wacc_threshold) & (cond_df[primary_col] > fluency_threshold)
        hall_strict = (cond_df["wacc"] <= wacc_q25) & (cond_df[primary_col] >= fluency_median)
        per_df.loc[cond_mask, "hallucination_like"] = hall_standard.values
        per_df.loc[cond_mask, "hallucination_like_strict"] = hall_strict.values

    rates = per_df.groupby("model_name", sort=False)[["hallucination_like", "hallucination_like_strict"]].mean()
    for row_idx, row in agg_df.iterrows():
        model_name = row["model_name"]
        if model_name in rates.index:
            agg_df.at[row_idx, "hallucination_like_rate"] = rates.loc[model_name, "hallucination_like"]
            agg_df.at[row_idx, "hallucination_like_rate_strict"] = rates.loc[model_name, "hallucination_like_strict"]
        agg_df.at[row_idx, "wacc_threshold"] = wacc_threshold
        agg_df.at[row_idx, "fluency_threshold"] = fluency_threshold

    return per_df, agg_df, {
        "wacc_threshold_mean_base": float(wacc_threshold),
        "fluency_threshold_mean_base": float(fluency_threshold),
        "wacc_q25_strict": float(wacc_q25),
        "fluency_median_strict": float(fluency_median),
    }


def compute_baseline_deltas(agg_df):
    base_row = agg_df[agg_df["condition"] == "base"]
    if len(base_row) == 0:
        print("  WARNING: no baseline row found; skipping deltas.", flush=True)
        return pd.DataFrame()

    base = base_row.iloc[0]
    delta_rows = []
    passthrough = {"condition", "model_name", "noise_condition", "noise_ratio", "n_samples"}
    for _, row in agg_df.iterrows():
        delta_row = {col: row[col] for col in passthrough if col in agg_df.columns}
        for col in agg_df.columns:
            if col in passthrough:
                continue
            base_val = base[col]
            curr_val = row[col]
            if pd.notna(base_val) and pd.notna(curr_val) and np.issubdtype(type(curr_val), np.number):
                delta_row[f"delta_{col}"] = curr_val - base_val
            else:
                try:
                    delta_row[f"delta_{col}"] = float(curr_val) - float(base_val)
                except (TypeError, ValueError):
                    delta_row[f"delta_{col}"] = np.nan
        delta_rows.append(delta_row)
    return pd.DataFrame(delta_rows)


def sort_input_paths(input_dir):
    paths = sorted(Path(input_dir).glob("results_*.tsv"))
    order = {name: idx for idx, name in enumerate(DEFAULT_CONDITION_ORDER)}
    return sorted(paths, key=lambda path: (order.get(parse_condition(path)[0], 999), path.name))


def resolve_device(requested_device):
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fairseq TSV outputs with Whisper-comparable metrics."
    )
    parser.add_argument("--input_dir", default="fairseq_results", help="Directory containing results_*.tsv files.")
    parser.add_argument("--output_dir", default="fairseq_eval", help="Directory for output CSV/JSON files.")
    parser.add_argument("--noise_ratio", type=float, default=0.08, help="Noise ratio for non-base fairseq runs.")
    parser.add_argument("--device", default="auto", help="Device for optional cosine/LM recomputation: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--recompute_cosine", action="store_true", help="Recompute cosine_similarity with sentence-transformers.")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for --recompute_cosine.")
    parser.add_argument("--recompute_lm", action="store_true", help="Recompute LM fluency instead of using scaled_probability_hypo.")
    parser.add_argument("--lm_models", nargs="+", default=["gpt2", "Qwen/Qwen3-0.6B"], help="Causal LMs for --recompute_lm.")
    parser.add_argument("--lm_batch_size", type=int, default=8, help="Batch size for --recompute_lm.")
    parser.add_argument("--primary_lm", default=None, help="Primary normalized_sentence_score_* suffix for hallucination rates.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    input_paths = sort_input_paths(args.input_dir)
    if not input_paths:
        raise FileNotFoundError(f"No results_*.tsv files found in {args.input_dir}")

    print(f"Found {len(input_paths)} fairseq result files:", flush=True)
    for path in input_paths:
        print(f"  {path}", flush=True)
    print(f"Using device: {args.device}", flush=True)

    per_df = build_per_utterance(input_paths, args)
    primary_lm, primary_col, lm_names = choose_primary_lm(per_df, requested_primary=args.primary_lm)
    print(f"\nPrimary fluency column: {primary_col}", flush=True)

    print("\nComputing aggregate metrics...", flush=True)
    agg_df = compute_aggregate_metrics(per_df, primary_col)

    print("Computing hallucination-like rates...", flush=True)
    per_df, agg_df, thresholds = compute_hallucination_rates(per_df, agg_df, primary_col)

    print("Computing baseline-relative deltas...", flush=True)
    delta_df = compute_baseline_deltas(agg_df)

    per_path = os.path.join(args.output_dir, "per_utterance_metrics_fairseq.csv")
    agg_path = os.path.join(args.output_dir, "aggregate_metrics_fairseq.csv")
    delta_path = os.path.join(args.output_dir, "baseline_relative_deltas_fairseq.csv")
    summary_path = os.path.join(args.output_dir, "summary_fairseq.json")

    per_df.to_csv(per_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "n_files": len(input_paths),
        "n_rows": int(len(per_df)),
        "models": sorted(per_df["model_name"].unique().tolist()),
        "noise_ratio_non_base": args.noise_ratio,
        "used_recomputed_cosine": args.recompute_cosine,
        "used_recomputed_lm": args.recompute_lm,
        "lm_names": lm_names,
        "primary_lm": primary_lm,
        "primary_fluency_column": primary_col,
        "thresholds": thresholds,
        "outputs": {
            "per_utterance": per_path,
            "aggregate": agg_path,
            "baseline_deltas": delta_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nWrote:", flush=True)
    print(f"  {per_path}", flush=True)
    print(f"  {agg_path}", flush=True)
    print(f"  {delta_path}", flush=True)
    print(f"  {summary_path}", flush=True)
    print("\nAggregate metrics:", flush=True)
    print(agg_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()