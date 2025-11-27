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

# Load environment variables from .env.local if it exists
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    echo "Loading environment variables from .env.local..."
    source "$PROJECT_ROOT/.env.local"
    # Explicitly export variables for child processes
    export HF_TOKEN
else
    echo "Warning: .env.local not found. HF_TOKEN may not be set."
fi

# Configure proxy settings for HuggingFace downloads
# IMPORTANT: If you're behind a proxy, uncomment and configure these variables:
# export HTTP_PROXY=http://172.17.0.1:1081
# export HTTPS_PROXY=http://172.17.0.1:1081
# export NO_PROXY=localhost,127.0.0.1
#
# NO_PROXY is crucial - it prevents proxying localhost connections,
# which would break health checks and benchmarking against local inference services

# Unset HF_HUB_OFFLINE to allow genai-bench to fetch tokenizer info from HuggingFace
unset HF_HUB_OFFLINE

# Start ARQ worker with nohup to survive session end
# Worker settings are in src/web/workers/autotuner_worker.py
# Use sg docker to ensure Docker socket access for Docker mode
echo "Starting ARQ worker with nohup..."
sg docker -c "nohup arq web.workers.autotuner_worker.WorkerSettings --verbose > logs/worker.log 2>&1 &"
sleep 1  # Give worker time to start
WORKER_PID=$(pgrep -f "arq web.workers.autotuner_worker.WorkerSettings")
echo "ARQ worker started with PID: $WORKER_PID"
echo $WORKER_PID > logs/worker.pid
echo "Worker logs: logs/worker.log"
