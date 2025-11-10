#!/bin/bash
# Start both web server and ARQ worker for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create logs directory if not exists
mkdir -p logs

# Activate virtual environment
source env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "🚀 Starting Inference Autotuner Development Environment..."
echo ""

# Start ARQ worker in background with nohup
echo "📋 Starting ARQ worker with nohup..."
nohup arq web.workers.autotuner_worker.WorkerSettings --verbose > logs/worker.log 2>&1 &
WORKER_PID=$!
echo $WORKER_PID > logs/worker.pid
echo "   Worker PID: $WORKER_PID"
echo "   Logs: logs/worker.log"
echo ""

# Wait a moment for worker to start
sleep 2

# Start web server (foreground)
echo "🌐 Starting web server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop web server (worker will continue running)"
echo "To stop worker: kill \$(cat logs/worker.pid)"
echo "---"

# Trap Ctrl+C to only kill web server (worker survives)
trap "echo ''; echo '🛑 Stopping web server...'; echo '   Worker still running with PID: $WORKER_PID'; exit" INT TERM

# Start web server in foreground
python src/web/server.py

echo ""
echo "Web server stopped. Worker still running with PID: $WORKER_PID"
