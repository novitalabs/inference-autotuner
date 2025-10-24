# Backend

Backend implementation for LLM Inference Autotuner web interface.

## Stack

- **FastAPI** - Web framework
- **SQLAlchemy** - ORM with async support
- **SQLite/aiosqlite** - Database
- **ARQ** - Task queue (Redis-based)
- **Pydantic** - Data validation

## Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── dev.py               # Development server runner
├── api/                 # API endpoints
│   ├── tasks.py         # Task management endpoints
│   ├── experiments.py   # Experiment endpoints
│   └── system.py        # System/health endpoints
├── core/                # Core configuration
│   └── config.py        # Settings management
├── db/                  # Database layer
│   ├── models.py        # SQLAlchemy models
│   └── session.py       # Database session
├── schemas/             # Pydantic schemas
│   └── __init__.py      # Request/response models
├── services/            # Business logic (future)
└── workers/             # ARQ background workers
    ├── autotuner_worker.py  # Main worker
    └── client.py        # ARQ client
```

## Setup

1. Install dependencies:
```bash
cd web/backend
pip install -r requirements.txt
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Start Redis (required for ARQ):
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally
redis-server
```

4. Run development server:
```bash
python dev.py
```

API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

5. Start ARQ worker (in separate terminal):
```bash
arq workers.autotuner_worker.WorkerSettings
```

## API Endpoints

### Tasks
- `POST /api/tasks` - Create new task
- `GET /api/tasks` - List all tasks
- `GET /api/tasks/{task_id}` - Get task by ID
- `GET /api/tasks/name/{task_name}` - Get task by name
- `PATCH /api/tasks/{task_id}` - Update task
- `DELETE /api/tasks/{task_id}` - Delete task
- `POST /api/tasks/{task_id}/start` - Start task execution
- `POST /api/tasks/{task_id}/cancel` - Cancel running task

### Experiments
- `GET /api/experiments/{experiment_id}` - Get experiment
- `GET /api/experiments/task/{task_id}` - List task experiments

### System
- `GET /health` - Health check
- `GET /api/system/health` - Detailed health check
- `GET /api/system/info` - System information

## Development

### Hot Reload
Development server (`dev.py`) has hot reload enabled. Changes to Python files will automatically restart the server.

### Database Migrations
Currently using SQLAlchemy's `create_all()` for simplicity. For production, consider using Alembic for migrations.

### Testing
```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=.
```

## Production

For production deployment:

1. Use PostgreSQL instead of SQLite:
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/autotuner
```

2. Use proper ASGI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

3. Run ARQ workers as separate processes:
```bash
arq workers.autotuner_worker.WorkerSettings --max-jobs 10
```

4. Use Redis cluster for high availability

## Notes

- Database file stored in `data/autotuner.db` (created automatically)
- ARQ requires Redis running
- Worker and API server must run simultaneously
- Default deployment mode is Docker (no Kubernetes required)
