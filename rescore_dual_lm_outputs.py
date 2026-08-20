#!/usr/bin/env python3
"""Attach explicit Qwen3 and GPT2 plausibility scores to saved ASR outputs.

This utility is intended for legacy stress files whose evaluator accepted
multiple LMs but wrote only the first LM to generic ``norm_plausibility``
columns. It avoids re-running Whisper: saved hypotheses are rescored directly.

For the project protocol, Qwen3-0.6B is the primary plausibility model and GPT2
is the parallel robustness model. Existing explicit per-LM columns are reused.
For legacy stress outputs generated with GPT2 as the first LM,
``norm_plausibility`` can be reused as GPT2 via ``--legacy_first_lm gpt2``.
Missing per-LM scores are computed with the same normalized hypothesis/reference
scoring helper used by the mitigation pipeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from mitigation_experiment import (
    DEFAULT_GPT2_MODEL,
    DEFAULT_QWEN_MODEL,
    _default_lm_plausibility,
)

QWEN_COL = "normalized_sentence_score_Qwen3-0.6B"
GPT2_COL = "normalized_sentence_score_gpt2"


def collect_paths(inputs: Sequence[str], globs: Sequence[str]) -> List[Path]:
    paths = [Path(p) for p in inputs]
    for pattern in globs:
        paths.extend(Path(p) for p in glob.glob(pattern))
    unique = sorted({p.resolve() for p in paths if p.exists()})
    if not unique:
        raise FileNotFoundError("No input files found for dual-LM rescoring")
    return unique


def read_table(path: Path) -> tuple[pd.DataFrame, str]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep), sep


def score_missing_rows(
    frames: Dict[Path, pd.DataFrame],
    *,
    column: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> None:
    refs: List[str] = []
    hyps: List[str] = []
    locations: List[tuple[Path, int]] = []

    for path, df in frames.items():
        if column not in df.columns:
            df[column] = np.nan
        missing = pd.to_numeric(df[column], errors="coerce").isna()
        for idx in df.index[missing]:
            refs.append(str(df.at[idx, "reference"]))
            hyps.append(str(df.at[idx, "hypothesis"]))
            locations.append((path, int(idx)))

    if not locations:
        print(f"{column}: all rows already scored; reusing existing values", flush=True)
        return

    print(
        f"{column}: scoring {len(locations):,} missing rows with {model_name}",
        flush=True,
    )
    scores = _default_lm_plausibility(
        hyps,
        refs,
        model_name,
        device,
        batch_size,
    )
    if len(scores) != len(locations):
        raise RuntimeError(
            f"LM scorer returned {len(scores)} scores for {len(locations)} rows"
        )
    for (path, idx), score in zip(locations, scores):
        frames[path].at[idx, column] = float(score)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="Input CSV/TSV; repeatable")
    p.add_argument("--input_glob", action="append", default=[], help="Input glob; repeatable")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--gpt2_model", default=DEFAULT_GPT2_MODEL)
    p.add_argument("--lm_batch_size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--legacy_first_lm",
        choices=["none", "gpt2", "qwen"],
        default="none",
        help=(
            "Interpret generic norm_plausibility as this LM when explicit columns are absent. "
            "Use gpt2 for the project's legacy stress TSVs."
        ),
    )
    args = p.parse_args()

    paths = collect_paths(args.input, args.input_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    frames: Dict[Path, pd.DataFrame] = {}
    separators: Dict[Path, str] = {}
    for path in paths:
        df, sep = read_table(path)
        missing_required = {"reference", "hypothesis"} - set(df.columns)
        if missing_required:
            raise ValueError(f"{path} missing required columns: {sorted(missing_required)}")

        # Normalize aliases from already-rescored/current evaluator outputs.
        if QWEN_COL not in df.columns:
            for candidate in ["qwen_plaus", "qwen_plausibility", "Qwen3"]:
                if candidate in df.columns:
                    df[QWEN_COL] = pd.to_numeric(df[candidate], errors="coerce")
                    break
        if GPT2_COL not in df.columns:
            for candidate in ["gpt2_plaus", "gpt2_plausibility"]:
                if candidate in df.columns:
                    df[GPT2_COL] = pd.to_numeric(df[candidate], errors="coerce")
                    break

        # Legacy evaluate_dual_metric.py wrote only its first LM to the generic
        # normalized column. Project stress scripts historically used GPT2 first.
        if "norm_plausibility" in df.columns:
            generic = pd.to_numeric(df["norm_plausibility"], errors="coerce")
            if args.legacy_first_lm == "gpt2":
                if GPT2_COL not in df.columns:
                    df[GPT2_COL] = generic
                else:
                    df[GPT2_COL] = pd.to_numeric(df[GPT2_COL], errors="coerce").fillna(generic)
            elif args.legacy_first_lm == "qwen":
                if QWEN_COL not in df.columns:
                    df[QWEN_COL] = generic
                else:
                    df[QWEN_COL] = pd.to_numeric(df[QWEN_COL], errors="coerce").fillna(generic)

        frames[path] = df
        separators[path] = sep

    score_missing_rows(
        frames,
        column=QWEN_COL,
        model_name=args.qwen_model,
        device=device,
        batch_size=args.lm_batch_size,
    )
    score_missing_rows(
        frames,
        column=GPT2_COL,
        model_name=args.gpt2_model,
        device=device,
        batch_size=args.lm_batch_size,
    )

    manifest = []
    for path in paths:
        df = frames[path]
        for col in [QWEN_COL, GPT2_COL]:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isna().any():
                raise RuntimeError(f"{path}: {int(values.isna().sum())} rows missing {col}")
            df[col] = values.astype(float)

        out_path = args.output_dir / path.name
        df.to_csv(out_path, sep=separators[path], index=False)
        manifest.append(
            {
                "source": str(path),
                "output": str(out_path),
                "rows": int(len(df)),
                "qwen_column": QWEN_COL,
                "gpt2_column": GPT2_COL,
            }
        )
        print(f"Saved dual-LM file: {out_path} ({len(df):,} rows)", flush=True)

    provenance = {
        "qwen_model": args.qwen_model,
        "gpt2_model": args.gpt2_model,
        "qwen_primary": True,
        "gpt2_parallel_robustness": True,
        "legacy_first_lm": args.legacy_first_lm,
        "files": manifest,
    }
    (args.output_dir / "dual_lm_rescoring_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
