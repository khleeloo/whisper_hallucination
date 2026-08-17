"""Build an acoustic-validation input CSV from TF-IDF candidate hypotheses.

The TF-IDF replication outputs retain `detail_row`, which corresponds to the row
index in the original evaluation detail TSV and Common Voice test manifest. This
script attaches the matching test audio path so wav2vec2 CTC scoring can be run
on those candidates.

Example:
    python prepare_tfidf_acoustic_input.py \
        --candidates_tsv tfidf_replication_clean_eval/candidate_hypotheses.tsv \
        --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
        --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
        --output_csv /scratch/vemotionsys/rmfrieske/whisper_hallucination/acoustic_validation/tfidf_candidate_acoustic_input.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_CANDIDATES = Path("tfidf_replication_clean_eval/candidate_hypotheses.tsv")
DEFAULT_TEST_TSV = Path("/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv")
DEFAULT_CLIPS_DIR = Path("/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips")
DEFAULT_OUTPUT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination/acoustic_validation/tfidf_candidate_acoustic_input.csv")


def load_manifest(test_tsv: Path, clips_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with test_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"Expected path column in {test_tsv}")
        for manifest_index, row in enumerate(reader):
            rel_path = row["path"]
            rows.append(
                {
                    "detail_row": manifest_index,
                    "audio_path": str(clips_dir / rel_path),
                    "manifest_path": rel_path,
                    "manifest_sentence": row.get("sentence", ""),
                    "manifest_sentence_id": row.get("sentence_id", ""),
                    "manifest_client_id": row.get("client_id", ""),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {test_tsv}")
    return pd.DataFrame(rows)


def comparable_text(value: object) -> str:
    return " ".join(str(value).lower().replace("'", " ").split())


def build_acoustic_input(candidates: pd.DataFrame, manifest: pd.DataFrame, max_mismatch_rate: float) -> pd.DataFrame:
    required = {"detail_row", "reference", "hypothesis", "wacc", "norm_plausibility", "eval_config", "perturbation"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    out = candidates.copy()
    out["detail_row"] = pd.to_numeric(out["detail_row"], errors="raise").astype(int)
    out = out.merge(manifest, on="detail_row", how="left", validate="many_to_one")
    missing_audio = out["audio_path"].isna().sum()
    if missing_audio:
        raise ValueError(f"Could not attach audio_path for {missing_audio} candidate rows")

    reference_norm = out["reference"].map(comparable_text)
    manifest_norm = out["manifest_sentence"].map(comparable_text)
    mismatches = int((reference_norm != manifest_norm).sum())
    mismatch_rate = float(mismatches / len(out)) if len(out) else 0.0
    if mismatch_rate > max_mismatch_rate:
        raise ValueError(
            f"Candidate/manifest alignment failed: {mismatches}/{len(out)} rows mismatch ({mismatch_rate:.1%}). "
            "The detail_row values do not appear to match this test manifest."
        )

    out["utterance_id"] = out.apply(lambda row: f"{row['eval_config']}_{row['perturbation']}_{int(row['detail_row']):06d}", axis=1)
    out["model_name"] = out["eval_config"].astype(str)
    out["condition"] = out["eval_config"].astype(str)
    out["perturbation_condition"] = out["perturbation"].astype(str)
    out["WER"] = 1.0 - pd.to_numeric(out["wacc"], errors="coerce")
    out["Qwen3_score"] = pd.to_numeric(out["norm_plausibility"], errors="coerce")
    out["GPT2_score"] = pd.to_numeric(out["norm_plausibility"], errors="coerce")
    out["tfidf_norm_plausibility_used_for_both_lm_scores"] = True

    rename_map = {
        "2gram_reps": "bigram_rep_count",
        "3gram_reps": "trigram_rep_count",
        "4gram_reps": "fourgram_rep_count",
    }
    out = out.rename(columns={old: new for old, new in rename_map.items() if old in out.columns})

    preferred = [
        "utterance_id", "audio_path", "reference", "hypothesis", "WER", "Qwen3_score", "GPT2_score",
        "model_name", "condition", "perturbation_condition", "eval_config", "perturbation", "detail_file",
        "detail_row", "analysis_id", "wacc", "norm_plausibility", "bigram_rep_count", "trigram_rep_count",
        "fourgram_rep_count", "manifest_path", "manifest_sentence_id", "manifest_client_id",
        "tfidf_norm_plausibility_used_for_both_lm_scores",
    ]
    columns = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach Common Voice audio paths to TF-IDF candidate hypotheses.")
    parser.add_argument("--candidates_tsv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--test_tsv", type=Path, default=DEFAULT_TEST_TSV)
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_mismatch_rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_tsv, sep="\t")
    manifest = load_manifest(args.test_tsv, args.clips_dir)
    out = build_acoustic_input(candidates, manifest, args.max_mismatch_rate)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out)} acoustic input rows to {args.output_csv}")


if __name__ == "__main__":
    main()
