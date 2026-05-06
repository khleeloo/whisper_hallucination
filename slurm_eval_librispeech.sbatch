#!/bin/bash
#SBATCH --job-name=wh_eval_ls
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval_ls.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_%a_eval_ls.err
#SBATCH --mem 200G
#SBATCH --array=0-4

# LibriSpeech cross-domain eval. Array index = config index (5 main configs).
# Each task evaluates one model on test-clean AND test-other (no perturbation).

module load Anaconda3
conda init
source activate llama

cd /home/rmfrieske/whisper_hallucination

CONFIGS=("base" "uu" "rr" "ru" "ur")
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

LS_ROOT=/scratch/vemotionsys/rmfrieske/datasets/librispeech_eval
LS_CLIPS=${LS_ROOT}/clips
MODEL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/${CONFIG}/final
EVAL_DIR=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results_librispeech

if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: model dir not found: ${MODEL_DIR}"
    exit 1
fi

for SPLIT in test_clean test_other; do
    echo "=== Eval ${CONFIG} on librispeech-${SPLIT} ==="
    python evaluate_dual_metric.py \
        --model_dir ${MODEL_DIR} \
        --base_model openai/whisper-large-v3 \
        --test_tsv ${LS_ROOT}/${SPLIT}.tsv \
        --clips_dir ${LS_CLIPS} \
        --output_dir ${EVAL_DIR}/${SPLIT} \
        --config_name ${CONFIG} \
        --lm_model Qwen/Qwen3-1.7B \
        --batch_size 8

    status=$?
    if [ $status -ne 0 ]; then
        echo "Eval FAILED for ${CONFIG} on ${SPLIT} (exit $status)"
        exit $status
    fi
done

echo "LibriSpeech eval complete for: ${CONFIG}"
