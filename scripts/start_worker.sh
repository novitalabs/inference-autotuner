#!/bin/bash
# Start ARQ worker for processing autotuning tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment
source env/bin/activate

# Add src directory to PYTHONPATH so imports work
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start ARQ worker
# Worker settings are in src/web/workers/autotuner_worker.py
echo "Starting ARQ worker..."
arq web.workers.autotuner_worker.WorkerSettings --verbose
