#!/bin/bash
#SBATCH --job-name=lora_mitig_smoke
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_lora_mitig_smoke.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_lora_mitig_smoke.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama
cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

python selective_lora_mitigation.py \
  --adapter /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-10000 \
  --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/selective_lora_mitigation_smoke \
  --max_samples 20 \
  --batch_size 4 \
  --skip_qwen \
  --seed 20260828
