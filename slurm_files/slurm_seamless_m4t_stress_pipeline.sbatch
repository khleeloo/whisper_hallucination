#!/bin/bash
#SBATCH --job-name=seamless_stress
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_seamless_stress.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_seamless_stress.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Keep the ~multi-GB Hugging Face model/cache off the home filesystem.
export HF_HOME=/scratch/vemotionsys/rmfrieske/huggingface_cache
mkdir -p "${HF_HOME}"

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

ROOT=/scratch/vemotionsys/rmfrieske/whisper_hallucination
OUTDIR=${ROOT}/seamless_m4t_v2_stress_pipeline
TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
MODEL=${SEAMLESS_MODEL:-facebook/seamless-m4t-v2-large}

mkdir -p "${OUTDIR}"

EXTRA_ARGS=()
if [[ -n "${DEV_TSV:-}" ]]; then
    EXTRA_ARGS+=(--dev_tsv "${DEV_TSV}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--resume)
fi
if [[ -n "${WAV2VEC2_MODEL:-}" ]]; then
    EXTRA_ARGS+=(--wav2vec2_model "${WAV2VEC2_MODEL}")
fi

cat <<EOF
=== Cross-architecture replication: SeamlessM4T-v2 Large ===
Host: $(hostname)
Job ID: ${SLURM_JOB_ID:-none}
Repo commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Model: ${MODEL}
Model loading: Transformers from_pretrained (automatic HF download/cache)
HF cache: ${HF_HOME}
DEV TSV: ${DEV_TSV:-auto-detect}
TEST TSV: ${TEST_TSV}
Output: ${OUTDIR}
Conditions: clean + full-noise 0.50 + full-noise 0.75
Hallucination evaluation: corrected WER + Qwen3/GPT-2
Strict criterion: WER > ${STRICT_WER:-0.5} + model-specific clean-DEV LM threshold
Gate calibration: stressed DEV, strict Qwen primary
Clean DEV coverage constraint: ${MIN_CLEAN_COVERAGE:-0.95}
TEST: frozen gate, no retuning
DEV samples/condition: ${DEV_MAX_SAMPLES:-1000}
TEST samples/condition: ${TEST_MAX_SAMPLES:-1000}
Seed: ${SEED:-20260820}
EOF

# Fail early with a useful message if this conda environment is too old.
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
try:
    import sentencepiece
    print("sentencepiece", sentencepiece.__version__)
except Exception as exc:
    raise SystemExit(f"sentencepiece is required for SeamlessM4T: {exc}")
try:
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    print("SeamlessM4Tv2ForSpeechToText available")
except Exception as exc:
    raise SystemExit(
        "Installed transformers does not support SeamlessM4Tv2ForSpeechToText. "
        f"Error: {exc}"
    )
PY

python -u seamless_m4t_stress_pipeline.py \
    --output_dir "${OUTDIR}" \
    --model "${MODEL}" \
    --tgt_lang eng \
    --test_tsv "${TEST_TSV}" \
    --clips_dir "${CLIPS_DIR}" \
    --dev_max_samples "${DEV_MAX_SAMPLES:-1000}" \
    --test_max_samples "${TEST_MAX_SAMPLES:-1000}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --lm_batch_size "${LM_BATCH_SIZE:-8}" \
    --ctc_batch_size "${CTC_BATCH_SIZE:-8}" \
    --strict_wer "${STRICT_WER:-0.5}" \
    --min_clean_coverage "${MIN_CLEAN_COVERAGE:-0.95}" \
    --seed "${SEED:-20260820}" \
    --qwen_model "${QWEN_MODEL:-Qwen/Qwen3-0.6B}" \
    --gpt2_model "${GPT2_MODEL:-gpt2}" \
    "${EXTRA_ARGS[@]}"

echo
echo "=== Done ==="
echo "Headline:           ${OUTDIR}/headline_summary.txt"
echo "TEST conditions:    ${OUTDIR}/test_condition_summary.csv"
echo "Frozen gate TEST:   ${OUTDIR}/test_gate_summary.csv"
echo "Strict clean cases: ${OUTDIR}/strict_clean_hallucination_candidates.csv"
echo "Gate threshold:     ${OUTDIR}/frozen_gate_threshold.json"
echo "H thresholds:       ${OUTDIR}/frozen_hallucination_thresholds.json"
echo "Per-output scores:  ${OUTDIR}/scored_outputs.csv"
echo "Report:             ${OUTDIR}/report.json"
