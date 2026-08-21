#!/bin/bash
#SBATCH --job-name=strict_clean_h
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_strict_clean_h.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_strict_clean_h.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
INPUT=${INPUT:-${ROOT}/clean_wer_rescore/scored_outputs_cleanwer.csv}
THRESHOLDS=${THRESHOLDS:-${ROOT}/clean_wer_rescore/frozen_hallucination_thresholds_cleanwer.json}
OUTDIR=${OUTDIR:-${ROOT}/clean_wer_rescore/strict_hallucinations}
WER_THRESHOLD=${WER_THRESHOLD:-0.50}
GATE_TAU=${GATE_TAU:-1.468906}

python -u strict_clean_hallucinations.py \
  --input "${INPUT}" \
  --thresholds "${THRESHOLDS}" \
  --output_dir "${OUTDIR}" \
  --wer_threshold "${WER_THRESHOLD}" \
  --gate_tau "${GATE_TAU}"
