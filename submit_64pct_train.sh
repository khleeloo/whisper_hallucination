#!/bin/bash
# Train 64% noise models: rr (Repeat-Repeat) and ru (Repeat-Unique)

sbatch \
    --job-name=wh64_train \
    --partition=normal \
    --account=vemotionsys \
    --nodes=1 \
    --cpus-per-task=16 \
    --gres=gpu:4 \
    --time=24:00:00 \
    --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_64_train.out \
    --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_64_train.err \
    --mem=256G \
    --array=0-1 \
    <<'SBATCH_SCRIPT'
#!/bin/bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("rr" "ru")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_64pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_64pct"

echo "=== Training ${CONFIG} with 64% noise on 4 GPUs ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# Use random port to avoid EADDRINUSE from stale torchrun processes on same node
MASTER_PORT=$((RANDOM + 30000))
torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} \
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
echo "=== Done: ${CONFIG}_64pct (exit code: $EXIT_CODE) ==="
SBATCH_SCRIPT

echo "64% training array jobs submitted."
