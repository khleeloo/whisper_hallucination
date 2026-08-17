#!/bin/bash
# Run the no-mitigation baseline for before/after mitigation experiments.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results/mitigation

python mitigation_experiment.py \
  --output_csv results/mitigation/baseline_outputs.csv \
  --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
  --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
  --conditions Base RR UR UU \
  "$@"
