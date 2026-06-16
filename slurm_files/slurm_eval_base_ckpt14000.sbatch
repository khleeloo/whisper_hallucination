#!/bin/bash
#SBATCH --job-name=wh_eval_base
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_eval_base_ckpt14000.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_eval_base_ckpt14000.err
#SBATCH --mem 64G

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

echo "=== Evaluating base checkpoint-14000 (best: eval_wer=7.54%, eval_loss=0.197) ==="

python evaluate_whisper_validation.py \
    --model_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/base/checkpoint-14000 \
    --base_model openai/whisper-large-v3 \
    --test_tsv /scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv \
    --clips_dir /scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips \
    --output_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation \
    --config_name base_checkpoint-14000 \
    --noise_condition base \
    --noise_ratio 0.08 \
    --batch_size 8 \
    --lm_models gpt2 Qwen/Qwen3-0.6B

echo "=== Done ==="
