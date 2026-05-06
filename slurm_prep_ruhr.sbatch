#!/bin/bash
#SBATCH --job-name=wh_prep_ruhr
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_prep_ruhr.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_prep_ruhr.err
#SBATCH --mem 32G

module load Anaconda3
conda init
source activate llama

cd /home/rmfrieske/whisper_hallucination

python create_noisy_dataset.py \
    --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
    --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
    --subset_size 120000 \
    --noise_ratio 0.08 \
    --seed 42

echo "RU/UR data prep complete"
