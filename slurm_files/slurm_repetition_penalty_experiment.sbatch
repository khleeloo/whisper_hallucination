#!/bin/bash
#SBATCH --job-name=wh_rep_pen
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_repetition_penalty.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_repetition_penalty.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p results/mitigation slurm_logs

TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
BASELINE_CSV=results/mitigation/baseline_outputs.csv
THRESHOLDS_JSON=results/mitigation/frozen_hallucination_thresholds.json
THRESHOLD_SOURCE_CSV=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/per_utterance_base_ckpt14000.csv
CTC_CACHE=results/mitigation/wav2vec2_ctc_score_cache.jsonl
SPLITS_CSV=results/mitigation/repetition_penalty_splits.csv
CONFIG_JSON=results/mitigation/repetition_penalty_config.json
COMMON_ARGS=(
    --baseline_csv "$BASELINE_CSV"
    --thresholds_json "$THRESHOLDS_JSON"
    --threshold_source_csv "$THRESHOLD_SOURCE_CSV"
    --test_tsv "$TEST_TSV"
    --clips_dir "$CLIPS_DIR"
    --ctc_cache_path "$CTC_CACHE"
    --conditions Base RR RU UR UU
    --batch_size 8
    --lm_batch_size 8
)
SPLIT_ARGS=(
    --splits_csv "$SPLITS_CSV"
    --dev_fraction 0.2
    --split_seed 1729
)
STATS_ARGS=(
    --bootstrap_resamples 10000
    --bootstrap_seed 1729
)

python -u repetition_penalty_experiment.py verify-1.00 \
    "${COMMON_ARGS[@]}" \
    --generated_csv results/mitigation/repetition_penalty_1.00_outputs.csv \
    --report_json results/mitigation/repetition_penalty_1.00_reproduction_report.json

python -u repetition_penalty_experiment.py dev-grid \
    "${COMMON_ARGS[@]}" \
    "${SPLIT_ARGS[@]}" \
    --penalties 1.00 1.05 1.10 1.15 1.20 \
    --outputs_csv results/mitigation/repetition_penalty_dev_outputs.csv \
    --grid_csv results/mitigation/repetition_penalty_dev_grid.csv

python -u repetition_penalty_experiment.py select \
    --baseline_csv "$BASELINE_CSV" \
    --splits_csv "$SPLITS_CSV" \
    --grid_csv results/mitigation/repetition_penalty_dev_grid.csv \
    --selection_json results/mitigation/repetition_penalty_selection.json \
    --config_json "$CONFIG_JSON"

python -u repetition_penalty_experiment.py clean-test \
    "${COMMON_ARGS[@]}" \
    "${SPLIT_ARGS[@]}" \
    "${STATS_ARGS[@]}" \
    --config_json "$CONFIG_JSON" \
    --outputs_csv results/mitigation/repetition_penalty_clean_test_outputs.csv \
    --before_after_csv results/mitigation/repetition_penalty_before_after.csv \
    --summary_csv results/mitigation/repetition_penalty_summary_by_condition.csv \
    --stats_csv results/mitigation/repetition_penalty_paired_stats.csv

python -u repetition_penalty_experiment.py perturbed-test \
    "${COMMON_ARGS[@]}" \
    "${SPLIT_ARGS[@]}" \
    "${STATS_ARGS[@]}" \
    --config_json "$CONFIG_JSON" \
    --perturbation_seed 8675309 \
    --before_after_csv results/mitigation/repetition_penalty_perturbed_before_after.csv \
    --summary_csv results/mitigation/repetition_penalty_perturbed_summary.csv \
    --stats_csv results/mitigation/repetition_penalty_perturbed_paired_stats.csv
