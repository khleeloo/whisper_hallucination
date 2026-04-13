"""
Aggregate and compare results across all noise configs and perturbations.
Produces summary tables matching the paper's format.

Usage:
    python aggregate_results.py \
        --results_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    # Load all result JSON files
    result_files = sorted(glob.glob(os.path.join(args.results_dir, "results_*.json")))

    if not result_files:
        print(f"No result files found in {args.results_dir}")
        return

    results = []
    for f in result_files:
        with open(f) as fh:
            results.append(json.load(fh))

    df = pd.DataFrame(results)

    # --- Table 1: Main results (no perturbation) ---
    print("=" * 70)
    print("TABLE 1: Main Evaluation Results (No Perturbation)")
    print("=" * 70)
    no_perturb = df[df["perturbation"] == "none"].copy()
    if not no_perturb.empty:
        no_perturb = no_perturb.set_index("config")
        no_perturb = no_perturb.reindex(["base", "uu", "ur", "rr", "ru"])
        cols = ["wacc", "avg_normalized_plausibility",
                "sentences_with_bigram_repeats", "sentences_with_trigram_repeats",
                "sentences_with_4gram_repeats"]
        display_names = {
            "wacc": "WAcc",
            "avg_normalized_plausibility": "Plausibility",
            "sentences_with_bigram_repeats": "Bigram Reps",
            "sentences_with_trigram_repeats": "Trigram Reps",
            "sentences_with_4gram_repeats": "4-gram Reps",
        }
        table1 = no_perturb[cols].rename(columns=display_names)
        print(table1.to_string())
        print()

    # --- Table 2: Perturbation impact ---
    print("=" * 70)
    print("TABLE 2: WAcc Under Perturbations")
    print("=" * 70)
    pivot = df.pivot_table(index="config", columns="perturbation", values="wacc")
    if "none" in pivot.columns:
        # Calculate relative drops
        for col in pivot.columns:
            if col != "none":
                pivot[f"drop_{col}"] = ((pivot["none"] - pivot[col]) / pivot["none"]) * 100
    order = ["base", "uu", "ur", "rr", "ru"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])
    print(pivot.to_string(float_format="%.4f"))
    print()

    # --- Table 3: Plausibility under perturbation ---
    print("=" * 70)
    print("TABLE 3: Plausibility Under Perturbations")
    print("=" * 70)
    pivot_p = df.pivot_table(index="config", columns="perturbation", values="avg_normalized_plausibility")
    pivot_p = pivot_p.reindex([c for c in order if c in pivot_p.index])
    print(pivot_p.to_string(float_format="%.4f"))
    print()

    # --- Table 4: Repetitions under perturbation ---
    print("=" * 70)
    print("TABLE 4: Trigram Repetitions Under Perturbations")
    print("=" * 70)
    pivot_r = df.pivot_table(index="config", columns="perturbation", values="sentences_with_trigram_repeats")
    pivot_r = pivot_r.reindex([c for c in order if c in pivot_r.index])
    print(pivot_r.to_string(float_format="%.0f"))

    # Save CSV
    csv_path = os.path.join(args.results_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
