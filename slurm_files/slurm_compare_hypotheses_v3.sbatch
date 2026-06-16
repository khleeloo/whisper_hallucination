#!/bin/bash
#SBATCH --job-name=wh_compare
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_compare_v3.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_compare_v3.err
#SBATCH --mem 16G

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama
export PYTHONUNBUFFERED=1

python3 /tmp/compare_hypotheses_nopandas.py
