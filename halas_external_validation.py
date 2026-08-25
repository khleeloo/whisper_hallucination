#!/usr/bin/env python3
"""Validate paper failure diagnostics on the human-annotated HALAS benchmark.

No ASR inference is required. The script downloads HALAS from a pinned commit,
flattens its seven paper-facing ASR systems, and compares structural diagnostics
against human span labels on the official HALAS test split.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, roc_auc_score

HALAS_COMMIT = "5c9c8b18fe67224dc10a801884cb5faa4d64b4fb"
HALAS_URL = (
    "https://raw.githubusercontent.com/DSP-AGH/HALAS/"
    f"{HALAS_COMMIT}/HALAS_dataset.csv"
)
MODELS = [
    "whisper_large_v2",
    "whisper_large_v3",
    "whisper_large_v3_turbo",
    "crisper_whisper",
    "canary",
    "canary_flash",
    "parakeet",
]
TOKEN_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def norm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return SPACE_RE.sub(" ", TOKEN_RE.sub(" ", str(x).lower())).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    if not r or norm(ref) == "inaudible":
        return np.nan
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def repetition(text):
    toks = norm(text).split()
    out = {}
    for n in (2, 3, 4):
        grams = [tuple(toks[i:i+n]) for i in range(max(0, len(toks) - n + 1))]
        counts = Counter(grams)
        out[f"rep{n}_count"] = sum(max(0, c - 1) for c in counts.values())
    out["rep34"] = int(out["rep3_count"] > 0 or out["rep4_count"] > 0)
    return out


def span_labels(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() in ("", "[]"):
        return False, False, False
    payload = json.loads(str(raw))
    h = l = lh = False
    for span in payload:
        labels = span.get("labels", []) if isinstance(span, dict) else []
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            name = str(label).lower()
            has_h = "halluc" in name
            has_l = "loop" in name
            if has_h and has_l:
                lh = True
            elif has_h:
                h = True
            elif has_l:
                l = True
    return h, l, lh


def phenotype(h, l, lh):
    if lh and (h or l):
        return "mixed_with_looping_hallucination"
    if lh:
        return "looping_hallucination"
    if h and l:
        return "mixed_hallucination_and_looping"
    if l:
        return "looping"
    if h:
        return "hallucination"
    return "none"


def get_reference(row):
    c = row.get("corrected_reference_text", "")
    if norm(c) and norm(c) != "inaudible":
        return str(c)
    return str(row.get("e22_reference_text", ""))


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "halas-validation/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, path.open("wb") as f:
        f.write(r.read())


def flatten(wide):
    rows = []
    for _, src in wide.iterrows():
        ref = get_reference(src)
        for model in MODELS:
            pred = src.get(f"{model}_prediction", "")
            if not norm(pred):
                continue
            h, l, lh = span_labels(src.get(f"{model}_hallucination_json", "[]"))
            label = src.get(f"{model}_label", "")
            label = "" if pd.isna(label) else str(label)
            pn, rn = norm(pred), norm(ref)
            rows.append({
                "audio_id": src.get("audio_id", ""),
                "split": src.get("split", ""),
                "model": model,
                "reference": ref,
                "prediction": pred,
                "prediction_norm": pn,
                "human_any_flag": bool(label and label.lower() != "no hallucination"),
                "human_hallucination": h or lh,
                "human_looping": l or lh,
                "human_pure_hallucination": h,
                "human_pure_looping": l,
                "human_looping_hallucination": lh,
                "phenotype": phenotype(h, l, lh),
                "WER": wer(ref, pred),
                "hyp_words": len(pn.split()),
                "ref_words": len(rn.split()),
                "length_ratio": len(pn.split()) / len(rn.split()) if rn.split() else np.nan,
                **repetition(pred),
            })
    return pd.DataFrame(rows)


def add_concentration(df):
    out = df.copy()
    out["same_hypothesis_frequency"] = 0
    out["top10_output_member"] = False
    out["top1_output_member"] = False
    for _, idx in out.groupby(["split", "model"], sort=False).groups.items():
        ids = list(idx)
        vals = out.loc[ids, "prediction_norm"]
        counts = vals.value_counts()
        top10, top1 = set(counts.head(10).index), set(counts.head(1).index)
        out.loc[ids, "same_hypothesis_frequency"] = [int(counts.get(v, 0)) for v in vals]
        out.loc[ids, "top10_output_member"] = [v in top10 for v in vals]
        out.loc[ids, "top1_output_member"] = [v in top1 for v in vals]
    out["duplicate_output"] = out["same_hypothesis_frequency"].astype(int) > 1
    return out


def fisher_assoc(df, predictor, target, split, name):
    g = df if split == "all" else df[df.split.eq(split)]
    p, t = g[predictor].astype(bool), g[target].astype(bool)
    a, b = int((p & t).sum()), int((p & ~t).sum())
    c, d = int((~p & t).sum()), int((~p & ~t).sum())
    odds, pval = fisher_exact([[a, b], [c, d]])
    return {
        "analysis": name, "split": split, "predictor": predictor, "target": target, "n": len(g),
        "predictor_rate_when_target_1": a / (a + c) if a + c else np.nan,
        "predictor_rate_when_target_0": b / (b + d) if b + d else np.nan,
        "target_rate_when_predictor_1": a / (a + b) if a + b else np.nan,
        "target_rate_when_predictor_0": c / (c + d) if c + d else np.nan,
        "odds_ratio": float(odds), "fisher_p": float(pval),
    }


def auc_row(g, score, target, name, split):
    mask = g[score].notna() & np.isfinite(pd.to_numeric(g[score], errors="coerce"))
    y, s = g.loc[mask, target].astype(int), pd.to_numeric(g.loc[mask, score], errors="coerce")
    roc = roc_auc_score(y, s) if y.nunique() == 2 else np.nan
    ap = average_precision_score(y, s) if y.nunique() == 2 else np.nan
    return {"analysis": name, "split": split, "predictor": score, "target": target,
            "n": len(y), "roc_auc": roc, "average_precision": ap}


def associations(df):
    rows = []
    for split in ("test", "all"):
        g = df if split == "all" else df[df.split.eq(split)]
        rows += [
            fisher_assoc(df, "rep34", "human_looping", split, "Rep34 vs human looping"),
            fisher_assoc(df, "top10_output_member", "human_hallucination", split,
                         "Top-10 recurring output vs human hallucination"),
            fisher_assoc(df, "duplicate_output", "human_hallucination", split,
                         "Any exact-output recurrence vs human hallucination"),
            auc_row(g, "WER", "human_hallucination", "WER discrimination of human hallucination", split),
            auc_row(g, "rep34", "human_looping", "Rep34 discrimination of human looping", split),
            auc_row(g, "same_hypothesis_frequency", "human_hallucination",
                    "Output recurrence discrimination of human hallucination", split),
        ]
    return pd.DataFrame(rows)


def phenotype_summary(df):
    rows = []
    for (split, ph), g in df.groupby(["split", "phenotype"], sort=False):
        rows.append({"split": split, "phenotype": ph, "N": len(g),
                     "rep34_rate": g.rep34.mean(),
                     "duplicate_output_rate": g.duplicate_output.mean(),
                     "top10_output_member_rate": g.top10_output_member.mean(),
                     "mean_same_hypothesis_frequency": g.same_hypothesis_frequency.mean(),
                     "mean_WER": g.WER.mean(), "median_WER": g.WER.median(),
                     "mean_length_ratio": g.length_ratio.mean()})
    return pd.DataFrame(rows)


def model_summary(df):
    rows = []
    for (split, model), g in df.groupby(["split", "model"], sort=False):
        counts = g.prediction_norm.value_counts()
        rows.append({"split": split, "model": model, "N": len(g),
                     "human_hallucination_rate": g.human_hallucination.mean(),
                     "human_looping_rate": g.human_looping.mean(), "rep34_rate": g.rep34.mean(),
                     "duplicate_output_rate": g.duplicate_output.mean(),
                     "top1_mass": counts.iloc[0] / len(g), "top10_mass": counts.iloc[:10].sum() / len(g),
                     "mean_WER": g.WER.mean()})
    return pd.DataFrame(rows)


def headline(df, assoc):
    t = df[df.split.eq("test")]
    loop, nloop = t[t.human_looping], t[~t.human_looping]
    hall, nhall = t[t.human_hallucination], t[~t.human_hallucination]
    rep = assoc[(assoc.split == "test") & (assoc.analysis == "Rep34 vs human looping")].iloc[0]
    top = assoc[(assoc.split == "test") &
                (assoc.analysis == "Top-10 recurring output vs human hallucination")].iloc[0]
    aucs = assoc[(assoc.split == "test") & assoc.roc_auc.notna()]
    lines = [
        "=== HALAS external validation: official test split ===",
        f"Predictions analyzed: {len(t):,} across {t.model.nunique()} ASR systems",
        f"Human hallucination rate: {t.human_hallucination.mean():.3f}",
        f"Human looping rate: {t.human_looping.mean():.3f}", "",
        "Structural phenotype validation:",
        f"  Rep34 among human-looping outputs: {loop.rep34.mean():.3f}",
        f"  Rep34 among non-looping outputs:   {nloop.rep34.mean():.3f}",
        f"  Fisher OR={rep.odds_ratio:.3f}, p={rep.fisher_p:.3g}",
        f"  Top-10 recurring-output membership among hallucinations: {hall.top10_output_member.mean():.3f}",
        f"  Top-10 recurring-output membership without hallucination: {nhall.top10_output_member.mean():.3f}",
        f"  Fisher OR={top.odds_ratio:.3f}, p={top.fisher_p:.3g}", "",
        "Single-axis discrimination (descriptive, not a proposed detector):",
    ]
    for _, r in aucs.iterrows():
        lines.append(f"  {r.analysis}: ROC-AUC={r.roc_auc:.3f}, AP={r.average_precision:.3f}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path)
    p.add_argument("--output_dir", type=Path, default=Path("halas_external_validation_outputs"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_csv or args.output_dir / "HALAS_dataset.csv"
    if args.input_csv is None:
        print(f"Downloading HALAS pinned at {HALAS_COMMIT}", flush=True)
        download(HALAS_URL, source)
    wide = pd.read_csv(source)
    long = add_concentration(flatten(wide))
    assoc = associations(long)
    ph = phenotype_summary(long)
    ms = model_summary(long)
    text = headline(long, assoc)

    long.to_csv(args.output_dir / "halas_per_prediction.csv", index=False)
    assoc.to_csv(args.output_dir / "halas_association_tests.csv", index=False)
    ph.to_csv(args.output_dir / "halas_phenotype_summary.csv", index=False)
    ms.to_csv(args.output_dir / "halas_model_summary.csv", index=False)
    (args.output_dir / "headline_summary.txt").write_text(text)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "dataset": "DSP-AGH/HALAS", "dataset_commit": HALAS_COMMIT,
        "models": MODELS, "audio_rows": len(wide), "predictions": len(long),
        "test_predictions": int(long.split.eq("test").sum()),
        "caution": "HALAS oversamples difficult/high-disagreement examples; rates are not prevalence estimates."
    }, indent=2) + "\n")
    print(text)
    print("Phenotypes (test):")
    print(ph[ph.split.eq("test")].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAssociations (test):")
    print(assoc[assoc.split.eq("test")].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
