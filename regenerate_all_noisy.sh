#!/bin/bash
# Regenerate ALL noisy datasets using SUBSTITUTION mode (same total size as clean baseline).
set -e

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en
OUTPUT_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
OUTPUT_DIR_24PCT=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct

cd /home/rmfrieske/whisper_hallucination

echo "=== Step 1: Main experiment datasets (uu, rr, ru, ur at 8%) ==="
python create_noisy_dataset.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --subset_size 120000 \
    --noise_ratio 0.08 \
    --seed 42

echo ""
echo "=== Step 2: Sweep datasets (all noise types x all ratios) ==="
python create_sweep_datasets.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --subset_size 120000 \
    --seed 42

echo ""
echo "=== Step 3: 24% special-case datasets (rr, ru, bare names) ==="
python create_sweep_datasets.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_24PCT" \
    --subset_size 120000 \
    --noise_types rr ru \
    --noise_ratios 0.24 \
    --name_template bare \
    --seed 42

echo ""
echo "=== Done! Verifying sizes... ==="
echo "Expected: all datasets should be ~120,000 rows"
