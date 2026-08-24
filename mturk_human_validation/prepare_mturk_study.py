#!/usr/bin/env python3
"""Prepare a blinded MTurk audio-grounding validation set for the ASR paper.

Uses only frozen paper-facing outputs and deterministically reconstructs the exact
clean/noisy waveform used by the acoustic-stress pipeline. It does not call
Mechanical Turk and does not upload audio.

Default study:
  - 180 experimental items (6 strata x 30)
  - 18 HIT rows, 10 experimental items + 1 QC item per HIT
  - publish each HIT later with 3 assignments
  - 6 reusable QC items (3 matched positive, 3 unrelated negative)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import torch
import torchaudio

from acoustic_abstention_mitigation import load_exact_waveform
from paper_final_audit import ROOT, add_final_wer, derive_thresholds, label, norm

MODEL_SOURCES = {
    "Raw Whisper": ROOT / "pretrained_whisper_stress_pipeline/rescore_explore/scored_outputs_corrected.csv",
    "Adapted Whisper": ROOT / "clean_wer_rescore/scored_outputs_cleanwer.csv",
    "SeamlessM4T-v2": ROOT / "seamless_m4t_v2_stress_pipeline_fixedwer/scored_outputs.csv",
}
SEVERE = ["full_noise_amp0.5_dur0.0", "full_noise_amp0.75_dur0.0"]
DEFAULT_OUTPUT = ROOT / "mturk_human_validation"
DEFAULT_SEED = 20260824
DEFAULT_NOISE_SEED = 20260821


def load_model_outputs(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing frozen source {path}")
    d = add_final_wer(pd.read_csv(path))
    th = derive_thresholds(d)
    d = label(d, th)
    if "split" in d.columns:
        d = d[d["split"].astype(str) == "test"].copy()
    d["model"] = name
    d["condition"] = d["perturbation"].astype(str)
    d["sample_key"] = (
        d["model"].astype(str) + "||" + d["condition"].astype(str) + "||" +
        d["utterance_id"].astype(str) + "||" + d["hypothesis"].astype(str)
    )
    return d.reset_index(drop=True)


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator, used: set[str]) -> pd.DataFrame:
    pool = pool[~pool["sample_key"].isin(used)].copy()
    if len(pool) < n:
        raise ValueError(f"Need {n} items but only {len(pool)} remain in candidate pool")
    idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    out = pool.loc[idx].copy()
    used.update(out["sample_key"].astype(str))
    return out


def _take_conditions(pool: pd.DataFrame, n: int, rng: np.random.Generator,
                     used: set[str], conditions: List[str]) -> pd.DataFrame:
    base = n // len(conditions)
    rem = n % len(conditions)
    pieces = []
    for i, cond in enumerate(conditions):
        want = base + (1 if i < rem else 0)
        pieces.append(_take(pool[pool["condition"] == cond], want, rng, used))
    return pd.concat(pieces, ignore_index=True)


def select_experimental_items(all_models: dict[str, pd.DataFrame], *,
                              n_per_stratum: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    selected = []

    # 1) Clean strict-H_Q candidates, balanced across models.
    per_model = n_per_stratum // 3
    remainder = n_per_stratum - 3 * per_model
    pieces = []
    for i, name in enumerate(all_models):
        n = per_model + (1 if i < remainder else 0)
        d = all_models[name]
        pool = d[(d["condition"] == "none") & d["strict_h_qwen_final"].astype(bool)]
        pieces.append(_take(pool, n, rng, used))
    x = pd.concat(pieces, ignore_index=True)
    x["stratum"] = "clean_strict_Hq"
    selected.append(x)

    # 2-4) Severe strict-H_Q, one stratum per model, balanced across 0.50/0.75.
    for name, slug in [
        ("Raw Whisper", "raw_severe_strict_Hq"),
        ("Adapted Whisper", "adapted_severe_strict_Hq"),
        ("SeamlessM4T-v2", "seamless_severe_strict_Hq"),
    ]:
        d = all_models[name]
        pool = d[d["condition"].isin(SEVERE) & d["strict_h_qwen_final"].astype(bool)]
        x = _take_conditions(pool, n_per_stratum, rng, used, SEVERE)
        x["stratum"] = slug
        selected.append(x)

    # 5) High-WER automatic non-H controls. Do not preselect by CTC support.
    pieces = []
    per_model = n_per_stratum // 3
    remainder = n_per_stratum - 3 * per_model
    for i, name in enumerate(all_models):
        n = per_model + (1 if i < remainder else 0)
        d = all_models[name]
        pool = d[
            d["condition"].isin(SEVERE)
            & (pd.to_numeric(d["WER_final"], errors="coerce") > 0.5)
            & (~d["strict_h_union_final"].astype(bool))
        ]
        pieces.append(_take_conditions(pool, n, rng, used, SEVERE))
    x = pd.concat(pieces, ignore_index=True)
    x["stratum"] = "high_WER_non_H_control"
    selected.append(x)

    # 6) Phenotype controls: half raw decoder-default collapse, half repetition.
    n_collapse = n_per_stratum // 2
    n_rep = n_per_stratum - n_collapse

    raw = all_models["Raw Whisper"].copy()
    rs = raw[raw["condition"].isin(SEVERE)].copy()
    rs["hyp_norm"] = rs["hypothesis"].map(norm)
    dominant = set(rs.loc[rs["hyp_norm"] != "", "hyp_norm"].value_counts().head(10).index)
    collapse_pool = rs[
        rs["hyp_norm"].isin(dominant)
        & (~rs["rep34_final"].astype(bool))
        & (~rs["strict_h_qwen_final"].astype(bool))
    ]
    collapse = _take(collapse_pool, n_collapse, rng, used)
    collapse["phenotype_seed"] = "decoder_default_collapse"

    n_adapt = n_rep // 2
    n_seam = n_rep - n_adapt
    rep_parts = []
    for name, n in [("Adapted Whisper", n_adapt), ("SeamlessM4T-v2", n_seam)]:
        d = all_models[name]
        pool = d[d["condition"].isin(SEVERE) & d["rep34_final"].astype(bool)]
        p = _take(pool, n, rng, used)
        p["phenotype_seed"] = "repetition_heavy"
        rep_parts.append(p)
    phenotype = pd.concat([collapse, *rep_parts], ignore_index=True)
    phenotype["stratum"] = "phenotype_control"
    selected.append(phenotype)

    out = pd.concat(selected, ignore_index=True)
    expected = 6 * n_per_stratum
    if len(out) != expected:
        raise AssertionError(f"Expected {expected} items, got {len(out)}")
    if out["sample_key"].duplicated().any():
        raise AssertionError("Duplicate experimental sample_key after selection")
    return out


def assign_blinded_ids(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    out = df.copy()
    ids = [f"mt_{i:06d}" for i in range(1, len(out) + 1)]
    rng.shuffle(ids)
    out["sample_id"] = ids
    out["audio_filename"] = out["sample_id"] + ".wav"
    return out


def export_waveform(row: object, path: Path, *, noise_seed: int) -> None:
    r = SimpleNamespace(audio_path=str(row.audio_path), split=str(row.split),
                        utterance_id=str(row.utterance_id))
    wav = load_exact_waveform(r, str(row.condition), base_seed=noise_seed)
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), tensor, 16000, encoding="PCM_S", bits_per_sample=16)


def _low_overlap(a: str, b: str) -> bool:
    aa, bb = set(norm(a).split()), set(norm(b).split())
    if not aa or not bb:
        return False
    return len(aa & bb) / max(1, len(aa | bb)) <= 0.10


def build_attention_checks(raw: pd.DataFrame, audio_dir: Path, *, seed: int,
                           noise_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 29)
    clean = raw[
        (raw["condition"] == "none")
        & np.isfinite(pd.to_numeric(raw["WER_final"], errors="coerce"))
        & (pd.to_numeric(raw["WER_final"], errors="coerce") == 0)
        & (raw["hypothesis"].astype(str).str.split().str.len() >= 4)
    ].copy()
    if len(clean) < 12:
        raise ValueError("Not enough exact clean rows to construct QC examples")
    chosen = clean.loc[rng.choice(clean.index.to_numpy(), size=6, replace=False)].reset_index(drop=True)
    rows = []

    for i in range(3):
        src = chosen.iloc[i]
        qid = f"qc_pos_{i+1:02d}"
        outpath = audio_dir / f"{qid}.wav"
        export_waveform(src, outpath, noise_seed=noise_seed)
        rows.append({"sample_id": qid, "audio_filename": outpath.name,
                     "display_hypothesis": str(src["hypothesis"]),
                     "expected_grounding": 3, "expected_class": "supported",
                     "source_utterance_id": src["utterance_id"],
                     "reference": src["reference"]})

    candidates = clean.reset_index(drop=True)
    for i in range(3):
        src = chosen.iloc[i + 3]
        alternatives = candidates[candidates["utterance_id"].astype(str) != str(src["utterance_id"])]
        alternatives = alternatives[
            alternatives["hypothesis"].map(lambda x: _low_overlap(str(src["reference"]), str(x)))
        ]
        if alternatives.empty:
            raise ValueError("Could not find a low-overlap transcript for negative QC")
        tgt = alternatives.iloc[int(rng.integers(0, len(alternatives)))]
        qid = f"qc_neg_{i+1:02d}"
        outpath = audio_dir / f"{qid}.wav"
        export_waveform(src, outpath, noise_seed=noise_seed)
        rows.append({"sample_id": qid, "audio_filename": outpath.name,
                     "display_hypothesis": str(tgt["hypothesis"]),
                     "expected_grounding": 0, "expected_class": "unsupported",
                     "source_utterance_id": src["utterance_id"],
                     "reference": src["reference"]})
    return pd.DataFrame(rows)


def make_hits(items: pd.DataFrame, qc: pd.DataFrame, *, items_per_hit: int,
              seed: int, audio_url_base: str) -> pd.DataFrame:
    if len(items) % items_per_hit != 0:
        raise ValueError(f"{len(items)} items is not divisible by items_per_hit={items_per_hit}")
    rng = random.Random(seed + 47)
    shuffled = items.sample(frac=1.0, random_state=seed + 47).reset_index(drop=True)
    n_hits = len(shuffled) // items_per_hit
    base = audio_url_base.rstrip("/") + "/"
    rows = []
    for h in range(n_hits):
        exp = shuffled.iloc[h * items_per_hit:(h + 1) * items_per_hit]
        q = qc.iloc[h % len(qc)]
        public = [{"sample_id": str(r.sample_id), "audio_filename": str(r.audio_filename),
                   "hypothesis": str(r.hypothesis)} for r in exp.itertuples(index=False)]
        public.append({"sample_id": str(q.sample_id), "audio_filename": str(q.audio_filename),
                       "hypothesis": str(q.display_hypothesis)})
        rng.shuffle(public)
        row = {"hit_batch_id": f"batch_{h+1:03d}"}
        for pos, item in enumerate(public, 1):
            k = f"{pos:02d}"
            row[f"sample_id_{k}"] = item["sample_id"]
            row[f"audio_filename_{k}"] = item["audio_filename"]
            row[f"audio_url_{k}"] = base + item["audio_filename"]
            row[f"hypothesis_{k}"] = item["hypothesis"]
        rows.append(row)
    return pd.DataFrame(rows)


def private_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "sample_id", "stratum", "phenotype_seed", "model", "condition", "split",
        "utterance_id", "audio_path", "reference", "hypothesis", "WER_final",
        "qwen_plaus", "gpt2_plaus", "diag_h_qwen_final", "diag_h_gpt2_final",
        "strict_h_qwen_final", "strict_h_gpt2_final", "strict_h_union_final",
        "rep34_final", "audio_filename", "sample_key",
    ]
    return [c for c in preferred if c in df.columns]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--noise_seed", type=int, default=DEFAULT_NOISE_SEED)
    p.add_argument("--n_per_stratum", type=int, default=30)
    p.add_argument("--items_per_hit", type=int, default=10)
    p.add_argument("--audio_url_base", default="https://REPLACE-ME.example/mturk_asr_audio")
    p.add_argument("--skip_audio", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_models = {name: load_model_outputs(name, path) for name, path in MODEL_SOURCES.items()}
    selected = assign_blinded_ids(
        select_experimental_items(all_models, n_per_stratum=args.n_per_stratum, seed=args.seed),
        args.seed,
    )

    if not args.skip_audio:
        for i, row in enumerate(selected.itertuples(index=False), 1):
            export_waveform(row, audio_dir / row.audio_filename, noise_seed=args.noise_seed)
            if i == 1 or i == len(selected) or i % 25 == 0:
                print(f"Exported experimental audio {i}/{len(selected)}", flush=True)

    qc = build_attention_checks(all_models["Raw Whisper"], audio_dir,
                                seed=args.seed, noise_seed=args.noise_seed)

    selected[private_columns(selected)].to_csv(
        args.output_dir / "private_sample_manifest.csv", index=False)
    qc.to_csv(args.output_dir / "attention_checks_PRIVATE.csv", index=False)

    public_items = selected[["sample_id", "audio_filename", "hypothesis"]].copy()
    public_items.rename(columns={"hypothesis": "display_hypothesis"}, inplace=True)
    public_items.to_csv(args.output_dir / "public_items.csv", index=False)

    hits = make_hits(selected, qc, items_per_hit=args.items_per_hit, seed=args.seed,
                     audio_url_base=args.audio_url_base)
    hits.to_csv(args.output_dir / "mturk_batch.csv", index=False)

    summary = {
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "n_experimental_items": int(len(selected)),
        "n_qc_items": int(len(qc)),
        "n_hits": int(len(hits)),
        "items_per_hit_experimental": args.items_per_hit,
        "items_per_hit_total": args.items_per_hit + 1,
        "recommended_assignments_per_hit": 3,
        "recommended_total_experimental_judgments": int(len(selected) * 3),
        "strata": selected["stratum"].value_counts().to_dict(),
        "models": selected["model"].value_counts().to_dict(),
        "conditions": selected["condition"].value_counts().to_dict(),
        "audio_url_base": args.audio_url_base,
        "source_files": {k: str(v) for k, v in MODEL_SOURCES.items()},
    }
    (args.output_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nPrepared MTurk study at: {args.output_dir}")
    print("PRIVATE: private_sample_manifest.csv, attention_checks_PRIVATE.csv")
    print("PUBLIC:  mturk_batch.csv")
    print("AUDIO:   audio/*.wav")


if __name__ == "__main__":
    main()
