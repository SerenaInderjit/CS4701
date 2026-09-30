#!/bin/bash

. .venv/bin/activate
export PYTHONPATH="$PWD"

case "$1" in
    test)
        pytest
        ;;
    play)
        python scripts/play_mario.py
        ;;
    eval)
        shift
        python scripts/evaluate_baseline.py "$@"
        ;;
    train)
        shift
        # Start dashboard if not already running
        if ! curl -s http://localhost:8000 > /dev/null 2>&1; then
            nohup .venv/bin/python -m uvicorn dashboard.server:app --port 8000 > /tmp/dashboard.log 2>&1 &
            sleep 2
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Dashboard: http://localhost:8000"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python scripts/train.py "$@"
        ;;
    *)
        echo "Usage: ./run.sh {test|play|eval [args]|train [args]}"
        exit 1
        ;;
esac
