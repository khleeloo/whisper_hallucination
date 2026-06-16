#!/bin/bash
#SBATCH --job-name=wh24_4gpu
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.err
#SBATCH --mem 256G
#SBATCH --array=0-1     # 0=rr, 1=ru
#SBATCH --dependency=afterok:449879

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("rr" "ru")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_24pct"

echo "=== Training ${CONFIG} with 24% noise on 4 GPUs ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# Use torchrun for multi-GPU DDP
# Use a random port in a wide range (20000-59999) to avoid collisions
# with other jobs on the same compute node or stale processes.
MASTER_PORT=$((20000 + RANDOM % 40000))

torchrun --nproc_per_node=4 --master_port="${MASTER_PORT}" \
    finetune_whisper_lora.py \
    --data_dir "${DATA_DIR}" \
    --noise_config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --clips_dir "${CLIPS_DIR}" \
    --model_name openai/whisper-large-v3 \
    --num_epochs 5 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete for ${CONFIG}"

    # Run eval immediately after training (single GPU)
    echo "=== Running eval for ${CONFIG} ==="
    EVAL_OUTPUT="/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_24pct"
    mkdir -p "${EVAL_OUTPUT}"

    python evaluate_whisper_validation.py \
        --model_dir "${OUTPUT_DIR}/final" \
        --base_model openai/whisper-large-v3 \
        --test_tsv "/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv" \
        --clips_dir "${CLIPS_DIR}" \
        --output_dir "${EVAL_OUTPUT}" \
        --config_name "${CONFIG}_24pct" \
        --noise_condition "${CONFIG}" \
        --noise_ratio 0.24 \
        --batch_size 8 \
        --lm_models Qwen/Qwen3-0.6B
fi

echo "=== Done: ${CONFIG} (exit code: $EXIT_CODE) ==="
