#!/bin/bash
#SBATCH --job-name=lora_mitig
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_lora_mitig.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_lora_mitig.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

python selective_lora_mitigation.py \
  --adapter /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-14000 \
  --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/selective_lora_mitigation \
  --max_samples 300 \
  --batch_size 8 \
  --qwen_batch_size 8 \
  --retain_gain 0.80 \
  --seed 20260828
