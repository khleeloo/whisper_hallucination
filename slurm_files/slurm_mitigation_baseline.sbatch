#!/bin/bash
#SBATCH --job-name=wh_mit_base
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_mitigation_baseline.out
#SBATCH --error=/home/rmfrieske/whisper_hallucination/slurm_logs/%j_mitigation_baseline.err

set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate llama

export PYTHONUNBUFFERED=1

cd /home/rmfrieske/whisper_hallucination
mkdir -p results/mitigation slurm_logs

TEST_TSV=/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv
CLIPS_DIR=/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips
THRESHOLDS_JSON=results/mitigation/frozen_hallucination_thresholds.json
THRESHOLD_SOURCE_CSV=/scratch/vemotionsys/rmfrieske/whisper_hallucination/eval_validation/per_utterance_base_ckpt14000.csv
CTC_CACHE=results/mitigation/wav2vec2_ctc_score_cache.jsonl
FINAL_OUTPUT=results/mitigation/baseline_outputs.csv
FINAL_REPORT=results/mitigation/baseline_reproduction_report.json
EXPECTED_ROWS=$(python - <<'PY'
import csv
import os

test_tsv = "/scratch/vemotionsys/rmfrieske/datasets/whisper_hallucination/test.tsv"
clips_dir = "/scratch/vemotionsys/rmfrieske/datasets/cv-corpus-22.0-2025-06-20/en/clips"
rows = 0
with open(test_tsv, "r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        if os.path.exists(os.path.join(clips_dir, row["path"])):
            rows += 1
print(rows)
PY
)
EXPECTED_LINES=$((EXPECTED_ROWS + 1))
CONDITIONS=(Base RU UU RR UR)
export EXPECTED_ROWS

is_complete_csv() {
    local csv_path=$1
    [[ -f "$csv_path" ]] || return 1
    CSV_PATH="$csv_path" python - <<'PY'
import os

import pandas as pd

csv_path = os.environ["CSV_PATH"]
expected_rows = int(os.environ["EXPECTED_ROWS"])
frame = pd.read_csv(csv_path)
if len(frame) != expected_rows:
    raise SystemExit(1)
if frame["condition"].nunique() != 1:
    raise SystemExit(1)
if set(frame["perturbation"].unique()) != {"none"}:
    raise SystemExit(1)
PY
}

for CONDITION in "${CONDITIONS[@]}"; do
    OUTPUT_CSV="results/mitigation/baseline_${CONDITION}_none.csv"
    REPORT_PATH="results/mitigation/baseline_${CONDITION}_none_report.json"
    LOG_PATH="results/mitigation/baseline_${CONDITION}_none.log"

    if is_complete_csv "$OUTPUT_CSV"; then
        echo "Skipping ${CONDITION}: complete output exists at ${OUTPUT_CSV}"
        continue
    fi

    echo "Running mitigation baseline for ${CONDITION}; writing ${OUTPUT_CSV}"
    python -u mitigation_experiment.py \
        --output_csv "$OUTPUT_CSV" \
        --report_path "$REPORT_PATH" \
        --thresholds_json "$THRESHOLDS_JSON" \
        --threshold_source_csv "$THRESHOLD_SOURCE_CSV" \
        --test_tsv "$TEST_TSV" \
        --clips_dir "$CLIPS_DIR" \
        --conditions "$CONDITION" \
        --perturbations none \
        --batch_size 8 \
        --lm_batch_size 8 \
        --ctc_cache_path "$CTC_CACHE" \
        2>&1 | tee "$LOG_PATH"

    if ! is_complete_csv "$OUTPUT_CSV"; then
        echo "Incomplete output after ${CONDITION}: expected ${EXPECTED_ROWS} CSV rows in ${OUTPUT_CSV}" >&2
        exit 1
    fi
done

python - <<'PY'
import os
from pathlib import Path

import pandas as pd

from mitigation_experiment import OUTPUT_COLUMNS

conditions = ["Base", "RU", "UU", "RR", "UR"]
expected_rows = int(os.environ["EXPECTED_ROWS"])
frames = []
for condition in conditions:
    csv_path = Path(f"results/mitigation/baseline_{condition}_none.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(f"{csv_path} has {len(frame)} rows; expected {expected_rows}")
    if set(frame["condition"].unique()) != {condition}:
        raise ValueError(f"{csv_path} does not contain exactly condition {condition}")
    if set(frame["perturbation"].unique()) != {"none"}:
        raise ValueError(f"{csv_path} does not contain only perturbation none")
    frames.append(frame[OUTPUT_COLUMNS])

output_path = Path("results/mitigation/baseline_outputs.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
print(f"Wrote {output_path} with {sum(len(frame) for frame in frames)} rows")
PY

python -u mitigation_experiment.py \
    --compare_only \
    --output_csv "$FINAL_OUTPUT" \
    --report_path "$FINAL_REPORT" \
    --reproduction_tolerance 0.02