#!/bin/bash
#SBATCH --job-name=wh64_uuur_eval
#SBATCH --partition=preempt
#SBATCH --account='vemotionsys'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH -o /home/rmfrieske/whisper_hallucination/slurm_logs/%j_64pct_uuur_eval.out
#SBATCH -e /home/rmfrieske/whisper_hallucination/slurm_logs/%j_64pct_uuur_eval.err
#SBATCH --mem 64G

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
TEST_TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS_DIR="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTPUT_DIR="${BASE_DIR}/eval_64pct"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/rmfrieske/whisper_hallucination/slurm_logs

CONFIGS=(uu_64pct ur_64pct)
CONDITIONS=(UU UR)

run_config() {
    local config="$1"
    local condition="$2"
    local model_dir="${BASE_DIR}/${config}"
    local ckpt_dir=""
    local ckpt_name=""

    if [ -d "${model_dir}/final" ]; then
        ckpt_dir="${model_dir}/final"
        ckpt_name="final"
    else
        while IFS= read -r candidate; do
            if [ -f "${candidate}/adapter_config.json" ] && \
               { [ -f "${candidate}/adapter_model.safetensors" ] || [ -f "${candidate}/adapter_model.bin" ]; }; then
                ckpt_dir="${candidate}"
                break
            fi
        done < <(find "${model_dir}" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' 2>/dev/null | sort -V -r)
        if [ -z "${ckpt_dir}" ]; then
            echo "ERROR: No complete adapter checkpoint found in ${model_dir}" >&2
            exit 1
        fi
        ckpt_name=$(basename "${ckpt_dir}")
    fi

    echo "=== Evaluating ${config}/${ckpt_name} ==="
    echo "Model dir: ${ckpt_dir}"
    echo "Output dir: ${OUTPUT_DIR}"

    run_shard() {
        local gpu_id="$1"
        local shard_id="$2"
        local suffix="$3"

        echo "=== Starting ${config}/${ckpt_name} shard ${shard_id}/2 on GPU ${gpu_id} ==="
        CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluate_whisper_validation.py \
            --model_dir "${ckpt_dir}" \
            --base_model openai/whisper-large-v3 \
            --test_tsv "${TEST_TSV}" \
            --clips_dir "${CLIPS_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --config_name "${config}_${ckpt_name}" \
            --noise_condition "${condition}" \
            --noise_ratio 0.64 \
            --batch_size 8 \
            --audio_num_workers 4 \
            --lm_batch_size 8 \
            --shard_id "${shard_id}" \
            --num_shards 2 \
            --output_suffix "${suffix}" \
            --lm_models gpt2 Qwen/Qwen3-0.6B
    }

    run_shard 0 0 "_shard00-of-02" &
    local pid0=$!
    run_shard 1 1 "_shard01-of-02" &
    local pid1=$!

    local status=0
    wait "${pid0}" || status=$?
    wait "${pid1}" || status=$?
    if [ "${status}" -ne 0 ]; then
        echo "ERROR: one or more shards failed for ${config} with status ${status}" >&2
        exit "${status}"
    fi

    echo "=== Done: ${config}/${ckpt_name} ==="
}

for index in "${!CONFIGS[@]}"; do
    run_config "${CONFIGS[$index]}" "${CONDITIONS[$index]}"
done

echo "=== All UU/UR 64pct evals complete ==="