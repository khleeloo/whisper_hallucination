#!/bin/bash
#SBATCH --job-name=wh_mit
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_mit.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_mit.err
#SBATCH --mem 200G
#SBATCH --array=0-1

# Mitigation experiment:
#   1) Filter the noisy training set with base Whisper (drop high-WER rows)
#   2) Re-train LoRA on the filtered set
# Targets the two configs hit hardest by label noise.
#
#   array idx 0 -> rr_10
#   array idx 1 -> uu_10

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("rr_10" "uu_10")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
FILTERED=${CONFIG}_filtered

DATA_DIR=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
SRC_TSV=${DATA_DIR}/${CONFIG}/train.tsv
DST_TSV=${DATA_DIR}/${FILTERED}/train.tsv
OUTPUT_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/${FILTERED}

export PYTHONUNBUFFERED=1

# Step 1: filter (skip if already filtered)
if [ ! -f "${DST_TSV}" ]; then
    echo "=== Filtering ${CONFIG} -> ${FILTERED} ==="
    python filter_training_data.py \
        --src_tsv ${SRC_TSV} \
        --dst_tsv ${DST_TSV} \
        --clips_dir ${CLIPS_DIR} \
        --wer_threshold 0.6 \
        --batch_size 16
    status=$?
    if [ $status -ne 0 ]; then
        echo "Filtering FAILED for ${CONFIG} (exit $status)"; exit $status
    fi
else
    echo "Filtered TSV already exists: ${DST_TSV}"
fi

# Step 2: train on filtered data
echo "=== Training on ${FILTERED} ==="
python finetune_whisper_lora.py \
    --noise_config ${FILTERED} \
    --data_dir ${DATA_DIR} \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_name openai/whisper-large-v3 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 500 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --seed 42

status=$?
if [ $status -ne 0 ]; then
    echo "Mitigation training FAILED for ${FILTERED} (exit $status)"; exit $status
fi
echo "Mitigation complete for: ${FILTERED}"
