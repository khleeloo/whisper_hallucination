#!/bin/bash
#SBATCH --job-name=halas_ft_hall
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_halas_ft_hall.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_halas_ft_hall.err

set -euo pipefail
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

python halas_finetune_forgetting.py generate \
  --adapter /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-10000 \
  --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/halas_finetune_forgetting \
  --sample_size 300 \
  --batch_size 8 \
  --seed 20260828
