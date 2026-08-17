#!/usr/bin/env python3
"""Create robustness table for hallucination-like criteria.

Rates are percentages of all utterances. Outputs with trigram or four-gram
repetition are excluded from the hallucination-like numerator so that
repetition/oscillation remains a separate failure category.
"""

from pathlib import Path

import pandas as pd


SCRATCH_ROOT = Path("/scratch/vemotionsys/rmfrieske/whisper_hallucination")
WORKSPACE = Path("/home/rmfrieske/whisper_hallucination")

BASE_FILE = SCRATCH_ROOT / "eval_validation" / "per_utterance_base_ckpt14000.csv"
EVAL_64PCT = SCRATCH_ROOT / "eval_64pct"
OUTPUT_TSV = WORKSPACE / "plots" / "hallucination_robustness_criteria.tsv"
OUTPUT_TEX = WORKSPACE / "hallucination_robustness_criteria_table.tex"

MODEL_FILES = {
    "Base": [BASE_FILE],
    "RR": [EVAL_64PCT / "per_utterance_rr_64pct_checkpoint-9375.csv"],
    "RU": [EVAL_64PCT / "per_utterance_ru_64pct_checkpoint-9375.csv"],
    "UR": [
        EVAL_64PCT / "per_utterance_ur_64pct_checkpoint-10000_shard00-of-02.csv",
        EVAL_64PCT / "per_utterance_ur_64pct_checkpoint-10000_shard01-of-02.csv",
    ],
    "UU": [
        EVAL_64PCT / "per_utterance_uu_64pct_final_shard00-of-02.csv",
        EVAL_64PCT / "per_utterance_uu_64pct_final_shard01-of-02.csv",
    ],
}

ORDER = ["Base", "RR", "RU", "UR", "UU"]


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def non_repeated_mask(df: pd.DataFrame) -> pd.Series:
    return (df["trigram_rep_count"].fillna(0).astype(float) == 0) & (
        df["fourgram_rep_count"].fillna(0).astype(float) == 0
    )


def compute_rates() -> pd.DataFrame:
    datasets = {condition: load_csvs(paths) for condition, paths in MODEL_FILES.items()}
    base = datasets["Base"]

    base_mean_wer = base["wer"].mean()
    base_mean_qwen = base["normalized_sentence_score_Qwen3-0.6B"].mean()
    base_wer_q75 = base["wer"].quantile(0.75)
    base_qwen_q75 = base["normalized_sentence_score_Qwen3-0.6B"].quantile(0.75)

    criteria = [
        (
            "Mean WER / mean Qwen3",
            lambda df: (df["wer"] > base_mean_wer)
            & (df["normalized_sentence_score_Qwen3-0.6B"] > base_mean_qwen),
        ),
        (
            "Top-25% WER / top-25% Qwen3",
            lambda df: (df["wer"] >= base_wer_q75)
            & (df["normalized_sentence_score_Qwen3-0.6B"] >= base_qwen_q75),
        ),
        (
            "WER > 0.5, Qwen3 > 0.6",
            lambda df: (df["wer"] > 0.5) & (df["normalized_sentence_score_Qwen3-0.6B"] > 0.6),
        ),
        (
            "WER > 0.5, Qwen3 > 0.7",
            lambda df: (df["wer"] > 0.5) & (df["normalized_sentence_score_Qwen3-0.6B"] > 0.7),
        ),
    ]

    rows = []
    for label, criterion in criteria:
        row = {"Criterion": label}
        for condition in ORDER:
            df = datasets[condition]
            mask = criterion(df) & non_repeated_mask(df)
            row[condition] = 100.0 * float(mask.mean())
            row[f"{condition}_count"] = int(mask.sum())
        rows.append(row)

    threshold_row = {
        "Criterion": "thresholds",
        "Base": f"mean_wer={base_mean_wer:.6f}; mean_qwen3={base_mean_qwen:.6f}; "
        f"wer_q75={base_wer_q75:.6f}; qwen3_q75={base_qwen_q75:.6f}",
    }
    rates = pd.DataFrame(rows)
    rates.attrs["thresholds"] = threshold_row["Base"]
    return rates


def write_latex(rates: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness of hallucination-like ranking under alternative criteria. Values are percentages of all utterances; samples with trigram or four-gram repetition are excluded from the hallucination-like numerator.}",
        r"\label{tab:hallucination_robustness}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Criterion} & \textbf{Base} & \textbf{RR} & \textbf{RU} & \textbf{UR} & \textbf{UU} \\",
        r"\midrule",
    ]
    for row in rates.itertuples(index=False):
        lines.append(
            f"{row.Criterion} & {row.Base:.2f} & {row.RR:.2f} & {row.RU:.2f} & {row.UR:.2f} & {row.UU:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5em}",
            r"\begin{minipage}{0.95\linewidth}",
            r"\footnotesize",
            r"\textit{Notes.} Hallucination-like outputs are treated as an operational diagnostic category: fluent or language-model-plausible hypotheses with low reference grounding. Repetition/oscillation is analyzed separately by excluding samples with elevated trigram or four-gram repetition from the hallucination-like numerator.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rates = compute_rates()
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.6f")
    write_latex(rates)

    print(rates[["Criterion", *ORDER]].to_csv(sep="\t", index=False, float_format="%.2f"), end="")
    print(f"\nSaved: {OUTPUT_TSV}")
    print(f"Saved: {OUTPUT_TEX}")
    print(f"Thresholds: {rates.attrs['thresholds']}")


if __name__ == "__main__":
    main()