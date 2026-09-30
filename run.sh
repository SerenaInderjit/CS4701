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
        python scripts/train.py "$@"
        ;;
    *)
        echo "Usage: ./run.sh {test|play|eval [args]|train [args]}"
        exit 1
        ;;
esac
