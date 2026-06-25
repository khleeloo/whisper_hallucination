#!/bin/bash
# Evaluate 64% RR/RU Whisper LoRA runs with explicit model/config matching.

set -euo pipefail

BASE_DIR="/scratch/vemotionsys/rmfrieske/whisper_hallucination"
EVAL_PY="/home/rmfrieske/whisper_hallucination/evaluate_whisper_validation.py"
BASE_MODEL="openai/whisper-large-v3"
TSV="/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
CLIPS="/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
OUTDIR="${BASE_DIR}/eval_64pct"

mkdir -p "${OUTDIR}" /home/rmfrieske/whisper_hallucination/slurm_logs

for config in rr_64pct ru_64pct; do
    condition="${config%%_*}"
    model_dir="${BASE_DIR}/${config}/final"

    if [ ! -f "${model_dir}/adapter_config.json" ]; then
        echo "ERROR: missing adapter_config.json for ${config}: ${model_dir}" >&2
        exit 1
    fi
    if [ ! -f "${model_dir}/adapter_model.safetensors" ] && [ ! -f "${model_dir}/adapter_model.bin" ]; then
        echo "ERROR: missing adapter weights for ${config}: ${model_dir}" >&2
        exit 1
    fi

    sbatch <<SBATCH
#!/bin/bash
#SBATCH --account=vemotionsys
#SBATCH --job-name=eval_${config}
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_${config}.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/eval_${config}.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=normal

set -euo pipefail

source /home/rmfrieske/.conda/envs/llama/etc/profile.d/conda.sh
conda activate llama

CUDA_VISIBLE_DEVICES=0 python "${EVAL_PY}" \
    --model_dir "${model_dir}" \
    --base_model "${BASE_MODEL}" \
    --test_tsv "${TSV}" \
    --clips_dir "${CLIPS}" \
    --output_dir "${OUTDIR}" \
    --config_name "${config}" \
    --noise_condition "${condition}" \
    --noise_ratio 0.64 \
    --batch_size 8 \
    --audio_num_workers 4 \
    --lm_batch_size 8 \
    --shard_id 0 \
    --num_shards 2 \
    --output_suffix _shard00-of-02 &
pid0=\$!

CUDA_VISIBLE_DEVICES=1 python "${EVAL_PY}" \
    --model_dir "${model_dir}" \
    --base_model "${BASE_MODEL}" \
    --test_tsv "${TSV}" \
    --clips_dir "${CLIPS}" \
    --output_dir "${OUTDIR}" \
    --config_name "${config}" \
    --noise_condition "${condition}" \
    --noise_ratio 0.64 \
    --batch_size 8 \
    --audio_num_workers 4 \
    --lm_batch_size 8 \
    --shard_id 1 \
    --num_shards 2 \
    --output_suffix _shard01-of-02 &
pid1=\$!

status=0
wait "\$pid0" || status=\$?
wait "\$pid1" || status=\$?
exit "\$status"
SBATCH
done

echo "64% two-GPU sharded eval jobs submitted to ${OUTDIR}."
