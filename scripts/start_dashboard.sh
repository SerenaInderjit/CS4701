#!/bin/bash
# Start the dashboard server in the background.
# Usage: ./scripts/start_dashboard.sh

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

# Kill any existing dashboard
kill $(lsof -ti:8000) 2>/dev/null
sleep 1

# Start dashboard with double-fork for full detachment on macOS
(nohup .venv/bin/python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8000 > /tmp/dashboard.log 2>&1 &)

sleep 3
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "Dashboard started: http://localhost:8000"
else
    echo "Dashboard failed to start. Check /tmp/dashboard.log"
    cat /tmp/dashboard.log
fi
