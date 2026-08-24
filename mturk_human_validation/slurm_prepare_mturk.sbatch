#!/bin/bash
#SBATCH --job-name=mturk_prepare
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_mturk_prepare.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_mturk_prepare.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
OUTDIR=${OUTDIR:-${ROOT}/mturk_human_validation}
SEED=${SEED:-20260824}
NOISE_SEED=${NOISE_SEED:-20260821}
AUDIO_URL_BASE=${AUDIO_URL_BASE:-https://REPLACE-ME.example/mturk_asr_audio}

cat <<EOF
=== MTurk human-validation preparation ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Output: ${OUTDIR}
Sampling seed: ${SEED}
Acoustic-noise seed: ${NOISE_SEED}
Audio URL base: ${AUDIO_URL_BASE}
Design: 180 experimental items; 10 experimental + 1 QC per HIT; recommend 3 assignments/HIT
EOF

python -u mturk_human_validation/prepare_mturk_study.py \
  --output_dir "${OUTDIR}" \
  --seed "${SEED}" \
  --noise_seed "${NOISE_SEED}" \
  --n_per_stratum 30 \
  --items_per_hit 10 \
  --audio_url_base "${AUDIO_URL_BASE}"

echo
echo "=== Done ==="
echo "PRIVATE manifest: ${OUTDIR}/private_sample_manifest.csv"
echo "PRIVATE QC key:   ${OUTDIR}/attention_checks_PRIVATE.csv"
echo "Public batch:     ${OUTDIR}/mturk_batch.csv"
echo "Audio directory:  ${OUTDIR}/audio"
echo "Study summary:    ${OUTDIR}/study_summary.json"
echo
echo "Do not upload the PRIVATE files to MTurk."
