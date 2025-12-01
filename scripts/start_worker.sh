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

# Load environment variables from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env..."
    # Export all variables from .env file
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "Warning: .env not found. Proxy and HF_TOKEN may not be set."
fi

# Load environment variables from .env.local if it exists (overrides .env)
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    echo "Loading environment variables from .env.local..."
    set -a
    source "$PROJECT_ROOT/.env.local"
    set +a
fi

# Verify proxy configuration
if [ -n "$HTTP_PROXY" ]; then
    echo "HTTP_PROXY configured: $HTTP_PROXY"
fi
if [ -n "$HTTPS_PROXY" ]; then
    echo "HTTPS_PROXY configured: $HTTPS_PROXY"
fi
if [ -n "$NO_PROXY" ]; then
    echo "NO_PROXY configured: $NO_PROXY"
fi

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
