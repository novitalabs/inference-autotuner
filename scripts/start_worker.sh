#!/bin/bash
# Start ARQ worker for processing autotuning tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create logs directory if not exists
mkdir -p logs

# Activate virtual environment
source env/bin/activate

# Add src directory to PYTHONPATH so imports work
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start ARQ worker with nohup to survive session end
# Worker settings are in src/web/workers/autotuner_worker.py
echo "Starting ARQ worker with nohup..."
nohup arq web.workers.autotuner_worker.WorkerSettings --verbose > logs/worker.log 2>&1 &
WORKER_PID=$!
echo "ARQ worker started with PID: $WORKER_PID"
echo $WORKER_PID > logs/worker.pid
echo "Worker logs: logs/worker.log"
