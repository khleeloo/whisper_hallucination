#!/bin/bash
#SBATCH --job-name=wh24_train
#SBATCH --partition=normal
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_train.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_train.err
#SBATCH --mem 256G

# Accept CONFIG as first argument: rr or ru
CONFIG=${1:-rr}

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

cd /home/rmfrieske/whisper_hallucination

DATA_DIR="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination_24pct"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}_24pct"

echo "=== Training ${CONFIG} with 24% noise on 4 GPUs ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# Step 1: Preprocess dataset ONCE using all CPUs (single process, no CUDA context)
# This avoids the DDP fork deadlock and runs much faster with num_proc=14
CACHE_FILE="${OUTPUT_DIR}/.dataset_cache_${CONFIG}_v2_done"
if [ ! -f "${CACHE_FILE}" ]; then
    echo "=== Preprocessing dataset (multi-proc, one-time) ==="
    python -c "
import os, sys, time, csv, multiprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WHISPER_SKIP_AUDIO_EXISTS_CHECK'] = '1'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
sys.path.insert(0, '/home/rmfrieske/whisper_hallucination')
from finetune_whisper_lora import (
    _build_cache_metadata,
    _write_cache_metadata,
    load_cv_dataset_from_tsv,
    prepare_dataset_batch,
)
from datasets import Dataset
try:
    from datasets import disable_progress_bar
except ImportError:
    from datasets.utils.logging import disable_progress_bar
from transformers import WhisperFeatureExtractor, WhisperTokenizer

disable_progress_bar()

feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-large-v3', language='en', task='transcribe')

train_tsv = '${DATA_DIR}/${CONFIG}/train.tsv'
dev_tsv = '${DATA_DIR}/dev.tsv'
clips_dir = '${CLIPS_DIR}'
cache_dir = '${OUTPUT_DIR}/.dataset_cache'
os.makedirs(cache_dir, exist_ok=True)

train_cache = os.path.join(cache_dir, 'train_${CONFIG}')
dev_cache = os.path.join(cache_dir, 'dev_${CONFIG}')

num_cpus = max(1, min(multiprocessing.cpu_count() - 2, 8))
print(f'Loading TSV data...', flush=True)
train_dataset = load_cv_dataset_from_tsv(train_tsv, clips_dir)
dev_dataset = load_cv_dataset_from_tsv(dev_tsv, clips_dir)
print(f'Preprocessing with {num_cpus} workers...', flush=True)
t0 = time.time()

train_dataset = train_dataset.map(
    lambda x: prepare_dataset_batch(x, feature_extractor, tokenizer),
    remove_columns=train_dataset.column_names,
    num_proc=num_cpus,
    batched=True,
    batch_size=16,
    writer_batch_size=1000,
)
dev_dataset = dev_dataset.map(
    lambda x: prepare_dataset_batch(x, feature_extractor, tokenizer),
    remove_columns=dev_dataset.column_names,
    num_proc=num_cpus,
    batched=True,
    batch_size=16,
    writer_batch_size=1000,
)

def filter_by_length(examples):
    return [length < 128 for length in examples['label_length']]
train_dataset = train_dataset.filter(filter_by_length, num_proc=num_cpus, batched=True, batch_size=1000, writer_batch_size=1000)
dev_dataset = dev_dataset.filter(filter_by_length, num_proc=num_cpus, batched=True, batch_size=1000, writer_batch_size=1000)

train_dataset.save_to_disk(train_cache)
dev_dataset.save_to_disk(dev_cache)
_write_cache_metadata(
    train_cache,
    _build_cache_metadata(train_tsv, dev_tsv, 'openai/whisper-large-v3', 128),
)
with open('${CACHE_FILE}', 'w') as f:
    f.write(f'Train: {len(train_dataset)}, Dev: {len(dev_dataset)}\n')
print(f'Cached {len(train_dataset)} train + {len(dev_dataset)} dev samples in {time.time()-t0:.0f}s', flush=True)
"
    echo "=== Preprocessing complete ==="
fi

# Step 2: Launch DDP training (loads cached Arrow dataset, instant startup)
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
    --skip_audio_exists_check \
    --preprocessing_batch_size 16 \
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
