#!/bin/bash
#SBATCH --job-name=failure_amp
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_failure_amp.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_failure_amp.err

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

OUT=/scratch/vemotionsys/rmfrieske/whisper_hallucination/failure_amplification
mkdir -p "$OUT"

python analyze_failure_amplification.py \
    --details_glob '/scratch/vemotionsys/rmfrieske/whisper_hallucination/stress_eval_64pct/details_*.tsv' \
    --details_glob '/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_64pct/details_*.tsv' \
    --output_dir "$OUT" \
    --bootstrap 10000 \
    --seed 42

echo ""
echo "=== Outputs ==="
ls -lh "$OUT"
