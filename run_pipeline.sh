#!/bin/bash
# Master script for Whisper hallucination experiments
# Run steps sequentially, each depending on the previous

set -e

cd /home/rmfrieske/whisper_hallucination
mkdir -p slurm_logs

echo "=== Whisper Hallucination Experiment Pipeline ==="
echo ""

# Step 1: Prepare noisy datasets
echo "Step 1: Preparing noisy datasets..."
PREP_JOB=$(sbatch --parsable slurm_prepare_data.sbatch)
echo "  Submitted prepare job: ${PREP_JOB}"

# Step 2: Train 5 models (base + 4 noise configs) — depends on step 1
echo "Step 2: Training 5 models (array job, depends on ${PREP_JOB})..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} slurm_train_array.sbatch)
echo "  Submitted training array job: ${TRAIN_JOB}"

# Step 3: Evaluate all models — depends on step 2
echo "Step 3: Evaluating all models (array job, depends on ${TRAIN_JOB})..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm_eval_array.sbatch)
echo "  Submitted evaluation array job: ${EVAL_JOB}"

echo ""
echo "Pipeline submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${PREP_JOB},${TRAIN_JOB},${EVAL_JOB}"
echo ""
echo "After completion, aggregate results with:"
echo "  python aggregate_results.py --results_dir /scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_results"
