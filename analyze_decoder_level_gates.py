#!/usr/bin/env python3
"""Analyze decoder-level gate families on saved high-noise detail TSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_full_noise_gated_acoustic_all"
)
DEFAULT_OUTPUT_DIR = Path("plots")


DETAIL_RE = re.compile(
    r"details_(?P<config>.+)_full_noise_amp(?P<amp>[^_]+)_dur0\.0_gated\.tsv$"
)

CONDITION_LABELS = {
    "base_checkpoint-10000": "Base",
    "rr_64pct_checkpoint-9375": "RR",
    "ru_64pct_checkpoint-9375": "RU",
    "uu_64pct_final": "UU",
    "ur_64pct_checkpoint-10000": "UR",
    "uu_100pct_final": "UU-100",
    "ur_100pct_checkpoint-4000": "UR-100",
}


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def summarize_flags(rows: list[dict[str, str]], flag_name: str, flags: list[bool]) -> dict[str, float]:
    hallucination = [parse_bool(row.get("hallucination_like", "0")) for row in rows]
    n_samples = len(rows)
    hallucination_count = sum(hallucination)
    flagged_count = sum(flags)
    true_positive = sum(flag and hall for flag, hall in zip(flags, hallucination))
    false_positive = flagged_count - true_positive
    true_negative_pool = n_samples - hallucination_count
    accepted = [not flag for flag in flags]
    accepted_count = sum(accepted)
    accepted_hallucination = sum(accept and hall for accept, hall in zip(accepted, hallucination))
    return {
        "signal": flag_name,
        "n_samples": n_samples,
        "hallucination_like_rate": safe_div(hallucination_count, n_samples),
        "flag_rate": safe_div(flagged_count, n_samples),
        "precision": safe_div(true_positive, flagged_count),
        "recall": safe_div(true_positive, hallucination_count),
        "false_positive_rate": safe_div(false_positive, true_negative_pool),
        "accepted_fraction": safe_div(accepted_count, n_samples),
        "accepted_hallucination_like_rate": safe_div(accepted_hallucination, accepted_count),
    }


def read_detail_file(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    match = DETAIL_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected detail filename: {path}")
    config = match.group("config")
    metadata = {
        "config": config,
        "condition": CONDITION_LABELS.get(config, config),
        "amp": match.group("amp"),
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return metadata, rows


def summarize_file(path: Path) -> list[dict[str, object]]:
    metadata, rows = read_detail_file(path)
    if not rows:
        return []

    has_token_path_alternatives = (
        "decoder_token_path_hashes" in rows[0]
        or "decoder_hypotheses" in rows[0]
    )
    low_confidence = [parse_bool(row.get("gate_avg_logprob_only", "0")) for row in rows]
    loop_collapse = [
        parse_bool(row.get("gate_ngram_repetition", "0"))
        or parse_bool(row.get("gate_compression_ratio_only", "0"))
        for row in rows
    ]
    token_path_disagreement = [
        parse_bool(row.get(
            "gate_token_path_disagreement_only",
            row.get("gate_hypothesis_disagreement_only", "0"),
        ))
        for row in rows
    ]
    beam_logprob_ambiguous = [
        parse_bool(row.get(
            "gate_beam_logprob_margin_only",
            row.get("gate_hypothesis_logprob_margin_only", "0"),
        ))
        for row in rows
    ]
    unstable_decoding = [parse_bool(row.get("gate_unstable_decoding", "0")) for row in rows]
    decoder_union = [
        low or loop or unstable
        for low, loop, unstable in zip(low_confidence, loop_collapse, unstable_decoding)
    ]

    summaries = []
    for signal_name, flags in [
        ("low_confidence", low_confidence),
        ("loop_collapse", loop_collapse),
        ("token_path_disagreement", token_path_disagreement),
        ("beam_logprob_margin_ambiguous", beam_logprob_ambiguous),
        ("unstable_decoding", unstable_decoding),
        ("decoder_union", decoder_union),
    ]:
        row = summarize_flags(rows, signal_name, flags)
        row.update(metadata)
        row["unstable_decoding_available"] = has_token_path_alternatives
        summaries.append(row)

    return summaries


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    metric_names = [
        "hallucination_like_rate",
        "flag_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "accepted_fraction",
        "accepted_hallucination_like_rate",
    ]
    out = []
    for key_values, group_rows in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, key_values)}
        row["n_runs"] = len(group_rows)
        row["n_samples"] = sum(int(group_row["n_samples"]) for group_row in group_rows)
        for metric in metric_names:
            row[metric] = sum(float(group_row[metric]) for group_row in group_rows) / len(group_rows)
        row["unstable_decoding_available"] = any(
            bool(group_row.get("unstable_decoding_available")) for group_row in group_rows
        )
        out.append(row)
    return out


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value: object) -> str:
    return f"{100.0 * float(value):5.1f}"


def print_condition_summary(rows: list[dict[str, object]]) -> None:
    print("condition\tsignal\thall_%\tflag_%\tprec_%\trecall_%\taccepted_hall_%")
    for row in rows:
        print(
            f"{row['condition']}\t{row['signal']}\t"
            f"{format_pct(row['hallucination_like_rate'])}\t"
            f"{format_pct(row['flag_rate'])}\t"
            f"{format_pct(row['precision'])}\t"
            f"{format_pct(row['recall'])}\t"
            f"{format_pct(row['accepted_hallucination_like_rate'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="details_*_full_noise_amp*_dur0.0_gated.tsv")
    args = parser.parse_args()

    detail_paths = sorted(args.input_dir.glob(args.pattern))
    if not detail_paths:
        raise SystemExit(f"No detail TSVs found in {args.input_dir}")

    by_run = []
    for path in detail_paths:
        by_run.extend(summarize_file(path))

    by_condition = aggregate(by_run, ["condition", "signal"])
    by_condition = [
        row for row in by_condition
        if not (row["condition"] in {"UU-100", "UR-100"} and row["signal"] == "unstable_decoding")
    ]

    run_path = args.output_dir / "decoder_level_gate_high_noise_by_run.tsv"
    condition_path = args.output_dir / "decoder_level_gate_high_noise_by_condition.tsv"
    write_tsv(run_path, by_run)
    write_tsv(condition_path, by_condition)

    print_condition_summary(by_condition)
    print(f"\nWrote {run_path}")
    print(f"Wrote {condition_path}")
    print(
        "\nNote: unstable_decoding is available only for TSVs produced with "
        "--num_decoder_hypotheses > 1. New files compare compact token-id paths; "
        "older single-hypothesis files report zero flags."
    )


if __name__ == "__main__":
    main()