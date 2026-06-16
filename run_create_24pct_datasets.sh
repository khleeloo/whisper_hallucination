#!/bin/bash
# Create 24% noise datasets using SUBSTITUTE mode (same total size as clean baseline).
# The slurm files sbatch_24pct_* expect rr and ur configs in whisper_hallucination_24pct/

cd /home/rmfrieske/whisper_hallucination

python create_sweep_datasets.py \
    --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
    --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct \
    --subset_size 120000 \
    --noise_types rr ru \
    --noise_ratios 0.24 \
    --name_template bare \
    --seed 42
