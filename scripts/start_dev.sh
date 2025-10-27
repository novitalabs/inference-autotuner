#!/bin/bash
# Start both web server and ARQ worker for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment
source env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "🚀 Starting Inference Autotuner Development Environment..."
echo ""

# Start ARQ worker in background
echo "📋 Starting ARQ worker..."
arq web.workers.autotuner_worker.WorkerSettings --verbose > logs/worker.log 2>&1 &
WORKER_PID=$!
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
echo "Press Ctrl+C to stop both services"
echo "---"

# Trap Ctrl+C to kill both processes
trap "echo ''; echo '🛑 Stopping services...'; kill $WORKER_PID 2>/dev/null; exit" INT TERM

# Start web server in foreground
python src/web/server.py

# Cleanup worker if server exits
kill $WORKER_PID 2>/dev/null
