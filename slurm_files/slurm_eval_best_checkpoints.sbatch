#!/bin/bash
#SBATCH --job-name=wh_eval_best
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval_best.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval_best.err
#SBATCH --mem 64G
#SBATCH --array=0-2
#  0=uu/checkpoint-6000  (best eval_wer=7.299%, epoch 0.74)
#  1=ur/checkpoint-14000 (best eval_wer=7.796%, epoch 1.73)
#  2=ru/checkpoint-10000 (best eval_wer=7.324%, epoch 1.23)

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation"

CONFIGS=("uu" "ur" "ru")
CKPTS=("checkpoint-6000" "checkpoint-14000" "checkpoint-10000")

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
MODEL_DIR="${BASE_DIR}/${CONFIG}/${CKPT}"

echo "=== Evaluating ${CONFIG}/${CKPT} ==="
echo "Model dir: ${MODEL_DIR}"

python evaluate_whisper_validation.py \
    --model_dir "${MODEL_DIR}" \
    --base_model openai/whisper-large-v3 \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --config_name "${CONFIG}_${CKPT}" \
    --noise_condition "${CONFIG}" \
    --noise_ratio 0.08 \
    --batch_size 8 \
    --lm_models gpt2 Qwen/Qwen3-0.6B

echo "=== Done: ${CONFIG}/${CKPT} ==="
