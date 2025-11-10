#!/bin/bash
# Stop ARQ worker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

PID_FILE="logs/worker.pid"

if [ -f "$PID_FILE" ]; then
    WORKER_PID=$(cat "$PID_FILE")
    if ps -p $WORKER_PID > /dev/null 2>&1; then
        echo "🛑 Stopping ARQ worker (PID: $WORKER_PID)..."
        kill $WORKER_PID
        sleep 1

        # Force kill if still running
        if ps -p $WORKER_PID > /dev/null 2>&1; then
            echo "   Force killing worker..."
            kill -9 $WORKER_PID
        fi

        rm -f "$PID_FILE"
        echo "✓ Worker stopped"
    else
        echo "⚠️  Worker (PID: $WORKER_PID) not running"
        rm -f "$PID_FILE"
    fi
else
    # Try to find worker by process name
    WORKER_PID=$(ps aux | grep "arq web.workers.autotuner_worker" | grep -v grep | awk '{print $2}')
    if [ -n "$WORKER_PID" ]; then
        echo "🛑 Found worker process (PID: $WORKER_PID), stopping..."
        kill $WORKER_PID
        echo "✓ Worker stopped"
    else
        echo "⚠️  No worker process found"
    fi
fi
