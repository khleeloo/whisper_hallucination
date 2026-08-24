#!/bin/bash
#SBATCH --job-name=paper_final_audit
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_paper_final_audit.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_paper_final_audit.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination

cat <<EOF
=== Final paper audit: tasks 1,2,3,5 ===
Host: $(hostname)
Job: ${SLURM_JOB_ID:-none}
Repo: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)

1) Final-WER rescore of controlled mismatch outputs
2) Cross-table/source consistency audit
3) Final-WER rescore of cached decision-utility outputs
5) 2,000-replicate bootstrap 95% CIs for cross-model headline metrics

No ASR or LM inference is rerun; cached outputs are reused.
GPU requested per cluster policy.
EOF

python -u paper_final_audit.py

echo
echo "=== Output files ==="
ls -lh "${ROOT}/paper_final_audit"
