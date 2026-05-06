#!/bin/bash
# Watchdog: submits remaining whisper_hallucination jobs as QOS slots free up.
# QOS limit = 10 jobs/user. Polls every 60s.
#
# Submission plan (priority order):
#   1. Sweep 7-11 (preempt, ~30-75 min each)        -> 5 array tasks
#   2. Main uu, rr, ru, ur (normal, --begin=+6h)    -> 4 jobs (array 1-4)
#   3. LibriSpeech prep (preempt, CPU)              -> 1 job
#   4. Seeds 0-9 (normal, --begin=+6h)              -> 10 array tasks
#   5. Mitigation 0-1 (normal, --begin=+6h)         -> 2 array tasks
#
# Each array submission counts as N pending jobs against the 10-slot QOS limit,
# so we only submit the next batch when (current_used + batch_size) <= 10.

set -u

cd /home/rmfrieske/whisper_hallucination
LOG=/home/rmfrieske/whisper_hallucination/slurm_logs/watchdog.log
USER=rmfrieske
QOS_MAX=10
RESERVE=0  # leave 0 slots free; submit aggressively

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_jobs() {
    # Count effective jobs in queue (array tasks count individually).
    squeue -u "$USER" -h -r | wc -l
}

slots_free() {
    local used
    used=$(count_jobs)
    echo $(( QOS_MAX - RESERVE - used ))
}

submit_when_room() {
    local needed=$1
    local desc=$2
    shift 2
    while true; do
        local free
        free=$(slots_free)
        if [ "$free" -ge "$needed" ]; then
            log "Submitting: $desc (needs $needed slots, $free free)"
            local out
            if out=$("$@" 2>&1); then
                log "  -> $out"
                return 0
            else
                log "  ERROR: $out"
                sleep 60
            fi
        else
            log "Waiting for $needed slots ($free free) before: $desc"
            sleep 60
        fi
    done
}

log "=== Watchdog started (PID $$) ==="

# PHASE 1 ONLY: sweeps + LibriSpeech prep.
# Main configs (uu/rr/ru/ur), seeds, and mitigation are DEFERRED until
# we evaluate the sweep and pick the optimal noise ratio.

# 1. Remaining sweep tasks 7-11
submit_when_room 5 "sweep 7-11" \
    sbatch --array=7-11 slurm_sweep_train.sbatch

# 2. LibriSpeech prep (CPU, fast)
submit_when_room 1 "LibriSpeech prep" \
    sbatch slurm_prepare_librispeech.sbatch

log "=== Watchdog finished (Phase 1 only). Run main/seeds/mitigation after picking noise ratio. ==="
