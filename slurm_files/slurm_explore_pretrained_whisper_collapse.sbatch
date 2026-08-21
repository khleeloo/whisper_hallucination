#!/bin/bash
#SBATCH --job-name=whisper_collapse
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_collapse.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_whisper_collapse.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
INPUT=${ROOT}/pretrained_whisper_stress_pipeline/scored_outputs.csv
OUTDIR=${ROOT}/pretrained_whisper_stress_pipeline/collapse_exploration

if [[ ! -f "${INPUT}" ]]; then
    echo "ERROR: missing ${INPUT}" >&2
    exit 1
fi

cat <<EOF
=== Explore untouched pretrained Whisper output collapse ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Input: ${INPUT}
Output: ${OUTDIR}
Top K: ${TOP_K:-20}
EOF

python -u explore_pretrained_whisper_collapse.py \
    --input "${INPUT}" \
    --output_dir "${OUTDIR}" \
    --top_k "${TOP_K:-20}"

echo
echo "=== Done ==="
echo "Report: ${OUTDIR}/collapse_report.txt"
echo "Top groups: ${OUTDIR}/top_hypotheses_by_condition.csv"
echo "Members: ${OUTDIR}/top_hypothesis_members.csv"
echo "Condition summary: ${OUTDIR}/collapse_condition_summary.csv"
