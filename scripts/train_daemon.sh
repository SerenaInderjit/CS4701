#!/bin/bash
# Daemonized training wrapper: survives shell restarts, auto-resumes on crash.
# Usage: ./scripts/train_daemon.sh --updates 1000 [--resume]

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

LOG_FILE="/tmp/train_daemon.log"
PID_FILE="/tmp/train_daemon.pid"
PYTHON=".venv/bin/python"

echo $$ > "$PID_FILE"
echo "$(date): Training daemon started (PID $$)" >> "$LOG_FILE"

run_training() {
    caffeinate -is "$PYTHON" scripts/train.py "$@" >> "$LOG_FILE" 2>&1
    return $?
}

# First attempt: resume if requested, otherwise fresh start
run_training "$@"
EXIT_CODE=$?

echo "$(date): Training exited with code $EXIT_CODE" >> "$LOG_FILE"

# Auto-resume on unexpected exit (not SIGTERM/SIGINT)
if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 143 ] && [ $EXIT_CODE -ne 130 ]; then
    echo "$(date): Auto-resuming in 10s..." >> "$LOG_FILE"
    sleep 10
    # Strip --resume from args if present, then add it
    ARGS=()
    for arg in "$@"; do
        [ "$arg" = "--resume" ] || ARGS+=("$arg")
    done
    run_training "${ARGS[@]}" --resume
fi

echo "$(date): Training daemon finished" >> "$LOG_FILE"
