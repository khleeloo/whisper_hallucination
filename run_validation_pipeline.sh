#!/bin/bash
# Master driver for Whisper Cross-Model Validation Pipeline
#
# Steps:
#   1. Submit eval job (array: 5 models, GPU) — produces per-utterance CSVs
#   2. Submit analysis job (CPU, dependency) — produces all aggregate tables + plots
#
# Usage:
#   bash run_validation_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p slurm_logs

echo "=== Whisper Cross-Model Validation Pipeline ==="
echo ""

# Step 1: Evaluate all 5 models (array job with GPU)
echo "Step 1: Submitting evaluation job (array: base, uu, rr, ru, ur)..."
EVAL_JOB=$(sbatch --parsable slurm_eval_validation.sbatch)
echo "  Eval job ID: ${EVAL_JOB}"

# Step 2: Analyze results (CPU, depends on Step 1)
echo "Step 2: Submitting analysis job (depends on ${EVAL_JOB})..."
ANALYSIS_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} slurm_analysis_validation.sbatch)
echo "  Analysis job ID: ${ANALYSIS_JOB}"

echo ""
echo "Pipeline submitted!"
echo "  Monitor eval:      squeue -j ${EVAL_JOB}"
echo "  Monitor analysis:  squeue -j ${ANALYSIS_JOB}"
echo ""
echo "After completion, find outputs at:"
echo "  /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/"
echo "  /scratch/vemotionsys/rmfrieske/whisper_hallucination/plots_validation/"
