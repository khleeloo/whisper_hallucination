#!/bin/bash
# Clean corrupted checkpoints and retrain 24% models (rr_24pct, ru_24pct)

# Clean corrupted checkpoint-2000 dirs so training starts fresh (no broken resume)
for config in rr ru; do
    CKPT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${config}_24pct/checkpoint-2000"
    if [ -d "$CKPT_DIR" ]; then
        echo "Removing corrupted checkpoint: $CKPT_DIR"
        rm -rf "$CKPT_DIR"
    fi
done

# Submit the 24% training array job (0=rr, 1=ru), no dependency
sbatch \
    --job-name=wh24_retrain \
    --partition=normal \
    --account=vemotionsys \
    --nodes=1 \
    --cpus-per-task=16 \
    --gres=gpu:4 \
    --time=24:00:00 \
    --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.out \
    --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_train.err \
    --mem=256G \
    --array=0-1 \
    <<'SBATCH_SCRIPT'
#!/bin/bash
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
echo "=== Done: ${CONFIG} (exit code: $EXIT_CODE) ==="
SBATCH_SCRIPT

echo "24% retraining jobs submitted."
