#!/usr/bin/env python3
"""Fail-fast numerical audit for the compressed ICASSP manuscript.

Uses only cached/final outputs: no ASR or LM inference. It verifies the three
paper tables, the clean-WER CI-overlap statement, CTC-gate coverage/capture
claims, the Raw-Whisper generic-output examples, and the qualitative GPT-2
separation.

Run:
    python paper_final_audit.py
    python paper_numerical_audit.py --tex paper_icassp.tex
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
FINAL = ROOT / "paper_final_audit"
REPORT = ROOT / "paper_numerical_audit.json"
SUMMARY = FINAL / "cross_model_summary_verified.csv"
CIS = FINAL / "cross_model_bootstrap95.csv"
UTILITY = FINAL / "decision_utility_final_rescore.csv"
RAW_GATE = ROOT / "pretrained_whisper_stress_pipeline/rescore_explore/test_gate_summary_corrected.csv"
SEAM_GATE = ROOT / "seamless_m4t_v2_stress_pipeline_fixedwer/test_gate_summary.csv"
RAW_TOP = ROOT / "pretrained_whisper_stress_pipeline/rescore_explore/top_hypotheses_by_condition.csv"

COND = {
    "none": "clean",
    "full_noise_amp0.5_dur0.0": ".50",
    "full_noise_amp0.75_dur0.0": ".75",
}
INV_COND = {v: k for k, v in COND.items()}
PAPER_MODEL = {
    "Raw Whisper": "Raw Whisper",
    "Adapted Whisper": "Adapted Whisper",
    "SeamlessM4T-v2": "SeamlessM4T",
}
INV_MODEL = {v: k for k, v in PAPER_MODEL.items()}


def need(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python paper_final_audit.py` first and make "
            "sure the final cached gate outputs exist."
        )


def block(tex: str, label: str) -> str:
    p = tex.find(f"\\label{{{label}}}")
    if p < 0:
        raise ValueError(f"Missing table label {label}")
    a = tex.rfind("\\begin{table}", 0, p)
    b = tex.find("\\end{table}", p)
    if a < 0 or b < 0:
        raise ValueError(f"Could not bound table {label}")
    return tex[a:b]


def cell(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\\\\s*$", "", s)
    s = s.replace("\\%", "").replace("$", "")
    s = s.replace("\\Delta", "Delta")
    return s.strip()


def number(s: str) -> float:
    s = cell(s).replace("+", "")
    s = re.sub(r"[^0-9eE.\-+]", "", s)
    return float(s) if s not in {"", "-", ".", "--"} else float("nan")


def same_round(a: float, b: float, d: int) -> bool:
    return math.isfinite(a) and math.isfinite(b) and round(a, d) == round(b, d)


def check(log: List[dict], name: str, ok: bool, expected=None, observed=None, source=None) -> None:
    log.append({
        "check": name,
        "ok": bool(ok),
        "expected": expected,
        "observed": observed,
        "source": source,
    })


def parse_stress(tex: str) -> Dict[Tuple[str, str], List[str]]:
    out: Dict[Tuple[str, str], List[str]] = {}
    current = None
    for line in block(tex, "tab:stress").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        c = [cell(x) for x in line.split("&")]
        if len(c) < 7 or c[0] == "Model":
            continue
        if c[0]:
            current = INV_MODEL.get(c[0])
        if current and c[1] in INV_COND:
            out[(current, INV_COND[c[1]])] = c[2:7]
    return out


def audit_stress(tex: str, df: pd.DataFrame, log: List[dict]) -> None:
    rows = parse_stress(tex)
    for r in df.itertuples():
        model, cond = str(r.model), str(r.condition)
        if model not in PAPER_MODEL or cond not in COND:
            continue
        vals = rows.get((model, cond))
        check(log, f"stress row {model} {COND[cond]}", vals is not None, "present", vals)
        if vals is None:
            continue
        specs = [
            (float(r.WER), number(vals[0]), 3, "WER"),
            (float(r.qwen_plaus), number(vals[1]), 3, "Qwen ratio"),
            (float(r.diag_H_qwen_pct), number(vals[2]), 1, "diag H_Q"),
        ]
        for a, b, d, m in specs:
            check(log, f"stress {model} {COND[cond]} {m}", same_round(a, b, d), round(a, d), b, str(SUMMARY))

        rep = float(r.rep34_pct)
        rep_txt = vals[3].replace(" ", "")
        if "<" in rep_txt:
            ok = model == "Adapted Whisper" and cond == "none" and rep < 0.5
            obs = rep_txt
        else:
            obs = number(vals[3])
            ok = same_round(rep, obs, 1)
        check(log, f"stress {model} {COND[cond]} Rep34", ok, round(rep, 1), obs, str(SUMMARY))

        if cond == "none":
            check(log, f"stress {model} clean Top-1 omitted", vals[4] == "--", "--", vals[4])
        else:
            obs = number(vals[4])
            check(log, f"stress {model} {COND[cond]} Top-1", same_round(float(r.top1_mass_pct), obs, 1), round(float(r.top1_mass_pct), 1), obs, str(SUMMARY))


def ci_range(s: str) -> Tuple[float, float]:
    m = re.search(r"([0-9.]+)\s*--\s*([0-9.]+)", cell(s))
    if not m:
        raise ValueError(f"Bad CI cell: {s!r}")
    return float(m.group(1)), float(m.group(2))


def audit_cis(tex: str, df: pd.DataFrame, log: List[dict]) -> None:
    aliases = {"Raw": "Raw Whisper", "Adapted": "Adapted Whisper", "Seamless": "SeamlessM4T-v2"}
    metrics = ["diag_H_qwen_pct", "rep34_pct", "top1_mass_pct"]
    for line in block(tex, "tab:ci").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        c = [cell(x) for x in line.split("&")]
        if len(c) < 5 or c[0] not in aliases or c[1] not in {".50", ".75"}:
            continue
        model, cond = aliases[c[0]], INV_COND[c[1]]
        for metric, txt in zip(metrics, c[2:5]):
            z = df[(df.model == model) & (df.condition == cond) & (df.metric == metric)]
            if len(z) != 1:
                check(log, f"CI source {model} {c[1]} {metric}", False, 1, len(z), str(CIS))
                continue
            lo, hi = float(z.iloc[0].ci_low), float(z.iloc[0].ci_high)
            plo, phi = ci_range(txt)
            check(log, f"CI {model} {c[1]} {metric}", round(lo, 1) == plo and round(hi, 1) == phi, [round(lo, 1), round(hi, 1)], [plo, phi], str(CIS))

    clean = df[(df.condition == "none") & (df.metric == "WER")]
    if len(clean) == 3:
        common = [float(clean.ci_low.max()), float(clean.ci_high.min())]
        check(log, "clean WER CIs overlap across all systems", common[0] <= common[1], "non-empty common overlap", common, str(CIS))
    else:
        check(log, "clean WER CI rows", False, 3, len(clean), str(CIS))


def utility_rows(tex: str) -> Dict[str, Tuple[float, float]]:
    out = {}
    for line in block(tex, "tab:utility").splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        c = [cell(x) for x in line.split("&")]
        if len(c) != 3 or c[0] == "Metric":
            continue
        a, b = number(c[1]), number(c[2])
        if math.isfinite(a) and math.isfinite(b):
            out[c[0]] = (a, b)
    return out


def find_row(rows: Dict[str, Tuple[float, float]], words: Tuple[str, ...]):
    for k, v in rows.items():
        low = k.lower()
        if all(w in low for w in words):
            return k, v
    return None, None


def audit_utility(tex: str, df: pd.DataFrame, log: List[dict]) -> None:
    rows = utility_rows(tex)
    cond = "full_noise_amp0.5_dur0.0"
    raw = df[(df.model == "Raw Whisper") & (df.condition == cond)].iloc[0]
    seam = df[(df.model == "SeamlessM4T-v2") & (df.condition == cond)].iloc[0]
    specs = [
        (("baseline", "wer"), "WER_baseline", 3),
        (("baseline", "strict"), "strict_H_qwen_baseline_pct", 1),
        (("baseline", "rep34"), "rep34_baseline_pct", 1),
        (("baseline", "top-1"), "top1_baseline_pct", 1),
        (("anti-rep", "wer"), "delta_WER", 3),
        (("anti-rep", "strict"), "delta_strict_H_qwen_pp", 1),
        (("anti-rep", "rep34"), "delta_rep34_pp", 1),
        (("anti-rep", "top-1"), "delta_top1_pp", 1),
        (("dominant-output", "abstention"), "collapse_abstention_pct", 1),
        (("strict", "captured"), "collapse_strict_H_qwen_capture_pct", 1),
    ]
    for words, col, d in specs:
        label, obs = find_row(rows, words)
        if obs is None:
            check(log, f"utility row {'/'.join(words)}", False, "present", None)
            continue
        exp = [float(raw[col]), float(seam[col])]
        ok = same_round(exp[0], obs[0], d) and same_round(exp[1], obs[1], d)
        check(log, f"utility {label}", ok, [round(x, d) for x in exp], list(obs), str(UTILITY))


def gate(path: Path, cond: str) -> pd.Series:
    d = pd.read_csv(path)
    key = "condition" if "condition" in d.columns else "perturbation"
    z = d[d[key].astype(str) == cond]
    if len(z) != 1:
        raise ValueError(f"Expected one {cond} row in {path}, got {len(z)}")
    return z.iloc[0]


def audit_ctc(tex: str, log: List[dict]) -> None:
    for p in (RAW_GATE, SEAM_GATE):
        need(p)
    rc, sc = gate(RAW_GATE, "none"), gate(SEAM_GATE, "none")
    clean_cov = [100 * float(rc.coverage), 100 * float(sc.coverage)]
    ok = all(f"{x:.1f}\\%" in tex for x in clean_cov)
    check(log, "CTC clean coverage prose", ok, [round(x, 1) for x in clean_cov], "paper text", f"{RAW_GATE}; {SEAM_GATE}")

    observed_cov = []
    captures = []
    for cond in ("full_noise_amp0.5_dur0.0", "full_noise_amp0.75_dur0.0"):
        for row in (gate(RAW_GATE, cond), gate(SEAM_GATE, cond)):
            observed_cov.append(round(100 * float(row.coverage), 1))
            captures.extend([float(row.strict_qwen_H_capture), float(row.strict_gpt2_H_capture)])
    check(log, "CTC severe coverage prose values", observed_cov == [0.1, 0.2, 0.0, 0.0], [0.1, 0.2, 0.0, 0.0], observed_cov, f"{RAW_GATE}; {SEAM_GATE}")
    check(log, "CTC rejects all strict Qwen/GPT2 cases under severe stress", all(np.isclose(x, 1.0) for x in captures), [1.0] * len(captures), captures, f"{RAW_GATE}; {SEAM_GATE}")


def audit_examples(log: List[dict]) -> None:
    need(RAW_TOP)
    d = pd.read_csv(RAW_TOP)
    z = d[d.condition.astype(str).isin(["full_noise_amp0.5_dur0.0", "full_noise_amp0.75_dur0.0"])]
    vals = set(z.normalized_hypothesis.fillna("").astype(str).str.strip().str.lower())
    for phrase in ("you", "thank you"):
        check(log, f"Raw severe top outputs contain '{phrase}'", phrase in vals, True, phrase in vals, str(RAW_TOP))


def audit_gpt2(df: pd.DataFrame, log: List[dict]) -> None:
    for cond in ("full_noise_amp0.5_dur0.0", "full_noise_amp0.75_dur0.0"):
        z = df[df.condition == cond]
        vals = dict(zip(z.model.astype(str), z.gpt2_plaus.astype(float)))
        raw = vals.get("Raw Whisper", float("inf"))
        ok = raw < vals.get("Adapted Whisper", float("-inf")) and raw < vals.get("SeamlessM4T-v2", float("-inf"))
        check(log, f"GPT-2 qualitative high-ratio separation {COND[cond]}", ok, "Raw < Adapted and Seamless", vals, str(SUMMARY))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", type=Path, default=REPO / "paper_icassp.tex")
    p.add_argument("--output", type=Path, default=REPORT)
    args = p.parse_args()
    for path in (args.tex, SUMMARY, CIS, UTILITY, RAW_GATE, SEAM_GATE, RAW_TOP):
        need(path)

    tex = args.tex.read_text(encoding="utf-8")
    summary = pd.read_csv(SUMMARY)
    cis = pd.read_csv(CIS)
    utility = pd.read_csv(UTILITY)
    log: List[dict] = []
    audit_stress(tex, summary, log)
    audit_cis(tex, cis, log)
    audit_utility(tex, utility, log)
    audit_ctc(tex, log)
    audit_examples(log)
    audit_gpt2(summary, log)

    failures = [x for x in log if not x["ok"]]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks_total": len(log),
        "checks_passed": len(log) - len(failures),
        "checks_failed": len(failures),
        "failures": failures,
        "checks": log,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== ICASSP PAPER NUMERICAL AUDIT ===")
    print(f"status: {result['status']}  passed: {result['checks_passed']}/{result['checks_total']}")
    for x in failures:
        print(f"FAIL: {x['check']} expected={x['expected']} observed={x['observed']}")
    print(f"Report: {args.output}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
