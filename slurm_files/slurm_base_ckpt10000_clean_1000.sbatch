#!/bin/bash
#SBATCH --job-name=base10k_clean
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_base10k_clean.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_base10k_clean.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
MODEL_DIR=${ROOT}/base/checkpoint-10000
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
OUTDIR=${ROOT}/failure_amplification
MAX_SAMPLES=1000

mkdir -p "${OUTDIR}"

echo "=== Matched clean Base evaluation ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: ${MODEL_DIR}"
echo "Test TSV: ${TEST_TSV}"
echo "Max samples: ${MAX_SAMPLES}"
echo "LMs: GPT2 + Qwen3-0.6B"
echo "Output: ${OUTDIR}"

if [[ ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
    echo "ERROR: missing adapter_config.json in ${MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/adapter_model.safetensors" && ! -f "${MODEL_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: missing adapter weights in ${MODEL_DIR}" >&2
    exit 1
fi

python -u evaluate_dual_metric.py \
    --model_dir "${MODEL_DIR}" \
    --base_model openai/whisper-large-v3 \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --output_dir "${OUTDIR}" \
    --config_name base \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size 8 \
    --lm_models gpt2 Qwen/Qwen3-0.6B \
    --perturb_type none \
    --perturb_amplitude 0.0 \
    --perturb_duration 0.0 \
    --num_log_examples 5

DETAILS="${OUTDIR}/details_base_none.tsv"
CLEAN_CSV="${OUTDIR}/per_utterance_base_ckpt10000_1000.csv"

if [[ ! -s "${DETAILS}" ]]; then
    echo "ERROR: expected details file missing: ${DETAILS}" >&2
    exit 1
fi

export DETAILS CLEAN_CSV TEST_TSV CLIPS_DIR MAX_SAMPLES

python - <<'PY'
import csv
import os
import re

import pandas as pd
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

details_path = os.environ["DETAILS"]
out_path = os.environ["CLEAN_CSV"]
test_tsv = os.environ["TEST_TSV"]
clips_dir = os.environ["CLIPS_DIR"]
max_samples = int(os.environ["MAX_SAMPLES"])

normalizer = BasicTextNormalizer()
cv_re = re.compile(r"(common_voice_en_\d+)", re.I)


def norm(x):
    return " ".join(normalizer(str(x)).split())


# Reproduce evaluate_dual_metric.load_test_data() exactly: max_samples is
# applied to the original TSV row index before missing-file filtering.
manifest = []
with open(test_tsv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        if i >= max_samples:
            break
        audio_path = os.path.join(clips_dir, row["path"])
        if not os.path.exists(audio_path):
            continue
        m = cv_re.search(row["path"])
        if not m:
            raise RuntimeError(
                f"Could not extract Common Voice ID from path: {row['path']}"
            )
        manifest.append({
            "utterance_id": m.group(1).lower(),
            "audio_path": audio_path,
            "manifest_reference": row["sentence"],
        })

details = pd.read_csv(details_path, sep="\t")
manifest = pd.DataFrame(manifest)

print(f"Evaluator detail rows: {len(details):,}")
print(f"Reconstructed manifest rows: {len(manifest):,}")

if len(details) != len(manifest):
    raise RuntimeError(
        f"Row-count mismatch: details={len(details)}, manifest={len(manifest)}"
    )

# Validate reconstructed row order after applying the same Whisper normalizer.
detail_refs = details["reference"].map(norm)
manifest_refs = manifest["manifest_reference"].map(norm)
bad = detail_refs != manifest_refs

if bad.any():
    examples = []
    for i in list(bad[bad].index[:10]):
        examples.append({
            "row": int(i),
            "details_reference": details.loc[i, "reference"],
            "manifest_reference": manifest.loc[i, "manifest_reference"],
            "details_normalized": detail_refs.loc[i],
            "manifest_normalized": manifest_refs.loc[i],
        })
    raise RuntimeError(
        f"{int(bad.sum())}/{len(details)} normalized reference mismatches. "
        f"Examples={examples}"
    )

# Add stable provenance columns expected by the failure-amplification analyzer.
details.insert(0, "utterance_id", manifest["utterance_id"])
details.insert(1, "audio_path", manifest["audio_path"])
details.to_csv(out_path, index=False)

print(f"Validated all {len(details):,} clean rows.")
print(f"Saved matched clean CSV: {out_path}")
print()
print(details[
    ["utterance_id", "audio_path", "reference", "hypothesis", "wer"]
].head().to_string(index=False))
PY

echo
echo "=== Matched Base clean evaluation complete ==="
echo "TSV: ${DETAILS}"
echo "CSV: ${CLEAN_CSV}"
