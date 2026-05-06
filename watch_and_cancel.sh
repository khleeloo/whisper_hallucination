#!/bin/bash
# Watch 372619_2_train.out and cancel the job if next eval_wer > BEST_WER
JOB_ID=372619
LOG=/home/rmfrieske/whisper_hallucination/slurm_logs/372619_2_train.out
BEST_WER=7.172122202336037
KNOWN_EVALS=5  # already seen 5 eval checkpoints (epochs 0.49,0.99,1.48,1.98,2.47,2.96 -> but 2.96 is not shown in checkpoint state let me use 6)

echo "Watching $LOG for next eval (job $JOB_ID, best WER so far: $BEST_WER)..."

PREV_COUNT=$(grep -c "eval_wer" "$LOG" 2>/dev/null || echo 0)

while true; do
    NEW_COUNT=$(grep -c "eval_wer" "$LOG" 2>/dev/null || echo 0)
    if [ "$NEW_COUNT" -gt "$PREV_COUNT" ]; then
        # New eval appeared
        LAST_WER=$(grep "eval_wer" "$LOG" | tail -1 | grep -oP "'eval_wer': \K[0-9.]+")
        LAST_EPOCH=$(grep "eval_wer" "$LOG" | tail -1 | grep -oP "'epoch': \K[0-9.]+")
        echo "[$(date)] New eval at epoch=$LAST_EPOCH: WER=$LAST_WER (best=$BEST_WER)"

        # Compare using python for float comparison
        IS_WORSE=$(python3 -c "print('yes' if $LAST_WER > $BEST_WER else 'no')")
        if [ "$IS_WORSE" = "yes" ]; then
            echo "WER $LAST_WER > best $BEST_WER — cancelling job $JOB_ID"
            scancel $JOB_ID
            echo "Job $JOB_ID cancelled."
            exit 0
        else
            echo "New best WER! Continuing..."
            BEST_WER=$LAST_WER
        fi
        PREV_COUNT=$NEW_COUNT
    fi
    sleep 60
done
