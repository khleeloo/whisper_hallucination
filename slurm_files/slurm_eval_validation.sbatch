#!/bin/bash
#SBATCH --job-name=wh_val_eval
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_val_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_val_eval.err
#SBATCH --mem 200G
#SBATCH --array=0-4

# Array job: 0=base, 1=uu, 2=rr, 3=ru, 4=ur
# Evaluates all 5 trained models with the full validation metric suite.

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "uu" "rr" "ru" "ur")
NOISE_CONDITIONS=("base" "UU" "RR" "RU" "UR")
NOISE_RATIOS=(0.0 0.08 0.08 0.08 0.08)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NOISE_COND=${NOISE_CONDITIONS[$SLURM_ARRAY_TASK_ID]}
NOISE_RATIO=${NOISE_RATIOS[$SLURM_ARRAY_TASK_ID]}

SCRATCH=/scratch/vemotionsys/rmfrieske/whisper_hallucination
MODEL_DIR=${SCRATCH}/${CONFIG}/final
DATA_DIR=${SCRATCH}/../datasets/whisper_hallucination
CLIPS_DIR=${SCRATCH}/../datasets/cv-corpus-22.0-2025-06-20/en/clips
TEST_TSV=${DATA_DIR}/test.tsv
OUTPUT_DIR=${SCRATCH}/eval_validation

mkdir -p ${OUTPUT_DIR}

echo "=== Validation Eval: ${CONFIG} (${NOISE_COND}) ==="
echo "Model dir: ${MODEL_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model dir not found: ${MODEL_DIR}"
    echo "Checking for checkpoint instead..."
    # Try latest complete numeric adapter checkpoint if final/ doesn't exist.
    LATEST_CKPT=""
    while IFS= read -r candidate; do
        if [ -f "${candidate}/adapter_config.json" ] && \
           { [ -f "${candidate}/adapter_model.safetensors" ] || [ -f "${candidate}/adapter_model.bin" ]; }; then
            LATEST_CKPT="${candidate}"
            break
        fi
    done < <(find "${SCRATCH}/${CONFIG}" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' 2>/dev/null | sort -V -r)
    if [ -n "${LATEST_CKPT}" ]; then
        echo "Using checkpoint: ${LATEST_CKPT}"
        MODEL_DIR=${LATEST_CKPT}
    else
        echo "FATAL: No model found for ${CONFIG}"
        exit 1
    fi
fi

# LMs to evaluate: GPT-2 (weak) + Qwen3 (strong) + Mistral if memory allows
# Start with the most important pair
python evaluate_whisper_validation.py \
    --model_dir ${MODEL_DIR} \
    --base_model openai/whisper-large-v3 \
    --test_tsv ${TEST_TSV} \
    --clips_dir ${CLIPS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --config_name ${CONFIG} \
    --noise_condition ${NOISE_COND} \
    --noise_ratio ${NOISE_RATIO} \
    --batch_size 8 \
    --lm_models gpt2 Qwen/Qwen3-1.7B

status=$?
if [ $status -ne 0 ]; then
    echo "Validation eval FAILED for ${CONFIG} (exit $status)"
    exit $status
fi

echo "Validation eval complete for: ${CONFIG}"
