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
    *)
        echo "Usage: ./run.sh {test|play}"
        exit 1
        ;;
esac