#!/bin/bash
#SBATCH --job-name=wh24_train
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.err
#SBATCH --mem 128G
#SBATCH --array=0-1

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("rr" "ru")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_24pct"

echo "=== Training ${CONFIG} with 24% noise ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

python finetune_whisper_lora.py \
    --data_dir "${DATA_DIR}" \
    --noise_config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --clips_dir "${CLIPS_DIR}" \
    --model_name openai/whisper-large-v3 \
    --num_epochs 5 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --seed 42

echo "Training complete for ${CONFIG}"

# Run eval immediately after training
echo "=== Running eval for ${CONFIG} ==="
EVAL_OUTPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_24pct"
mkdir -p "${EVAL_OUTPUT}"

python evaluate_whisper_validation.py \
    --model_dir "${OUTPUT_DIR}/final" \
    --base_model openai/whisper-large-v3 \
    --test_tsv "${DATA_DIR}/test.tsv" \
    --clips_dir "${CLIPS_DIR}" \
    --output_dir "${EVAL_OUTPUT}" \
    --config_name "${CONFIG}_24pct" \
    --noise_condition "${CONFIG}" \
    --noise_ratio 0.24 \
    --batch_size 8 \
    --lm_models Qwen/Qwen3-0.6B

echo "=== Done: ${CONFIG} ==="
