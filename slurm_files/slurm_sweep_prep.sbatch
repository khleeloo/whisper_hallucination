#!/bin/bash
#SBATCH --job-name=wh_sweep_prep
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_sweep_prep.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_sweep_prep.err
#SBATCH --mem 32G

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

python create_sweep_datasets.py \
    --data_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en \
    --output_dir /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination \
    --subset_size 120000 \
    --seed 42

echo "Sweep data preparation complete"
