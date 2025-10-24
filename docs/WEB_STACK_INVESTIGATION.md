# Web Application Technical Stack Investigation

**Project**: LLM Inference Autotuner
**Date**: October 24, 2025
**Purpose**: Determine best practice technical stack for web interface

---

## Project Requirements Analysis

### Core Requirements
1. **Long-running tasks** - Parameter tuning experiments (2-10 minutes each)
2. **Real-time progress** - Live updates during experiment execution
3. **Task queue** - Manage multiple tuning jobs (sequential or parallel)
4. **Results visualization** - Display metrics, charts, comparisons
5. **Resource management** - Track GPU usage, container status
6. **Multi-user support** - Multiple users submitting tasks (future)
7. **Data persistence** - Store tasks, experiments, results
8. **API-first design** - Programmatic access for automation

### Non-functional Requirements
- **Performance**: Handle concurrent users with long-running tasks
- **Reliability**: Don't lose task state on server restart
- **Scalability**: Scale from single-user to multi-user
- **Developer Experience**: Fast development, good tooling
- **Maintainability**: Clear code structure, good documentation

---

## Technical Stack Recommendations

## 🏆 RECOMMENDED STACK (Modern, Best Practice)

### Backend: **FastAPI** ✅

**Why FastAPI?**
- ✅ **Best Performance**: 70K+ GitHub stars, fastest Python framework
- ✅ **Async-Native**: Perfect for long-running tasks and WebSockets
- ✅ **Auto Documentation**: Swagger UI + ReDoc out-of-the-box
- ✅ **Type Safety**: Pydantic models with automatic validation
- ✅ **Modern Python**: Uses Python 3.8+ features (async/await, type hints)
- ✅ **Growing Ecosystem**: 2025 trend leader for AI/ML applications

**FastAPI Features:**
```python
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI()

class Task(BaseModel):
    task_name: str
    parameters: dict

@app.post("/api/tasks")
async def create_task(task: Task):
    # Automatic validation, serialization, docs
    return {"status": "created"}

@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    # Real-time updates
```

**Comparison with Alternatives:**
- **vs Django**: FastAPI is 3-5x faster, better async support, less boilerplate
- **vs Flask**: FastAPI has built-in validation, docs, async; Flask needs extensions
- **Performance**: FastAPI handles 10,000+ requests/sec (Django: 2,000, Flask: 3,000)

**Best for**: Microservices, APIs, async-heavy apps, AI/ML backends (2025 trend)

---

### Task Queue: **ARQ** (AsyncIO) or **Celery** (Production) ✅

#### Option A: **ARQ** (Recommended for MVP)

**Why ARQ?**
- ✅ **AsyncIO-Native**: Perfect match for FastAPI
- ✅ **Simple Setup**: Minimal configuration
- ✅ **Redis-Based**: Fast, reliable, easy to deploy
- ✅ **FastAPI Integration**: Built specifically for async frameworks
- ✅ **Lightweight**: No heavy dependencies

**ARQ Example:**
```python
from arq import create_pool
from arq.connections import RedisSettings

async def run_tuning_task(ctx, task_id: str):
    """Background task for autotuning"""
    orch = AutotunerOrchestrator(...)
    result = orch.run_task(task_file)
    return result

# FastAPI integration
@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str):
    await redis.enqueue_job('run_tuning_task', task_id)
    return {"status": "queued"}
```

**When to Use:**
- MVP/initial development
- < 10 concurrent background tasks
- FastAPI-based application
- Simple retry/timeout logic needed

#### Option B: **Celery** (Recommended for Production)

**Why Celery?**
- ✅ **Battle-Tested**: Most popular Python task queue (10+ years)
- ✅ **Feature-Rich**: Advanced retry, scheduling, monitoring (Flower UI)
- ✅ **Scalable**: Handles thousands of workers
- ✅ **Multiple Brokers**: Redis, RabbitMQ, SQS
- ✅ **Monitoring**: Built-in tools (Flower dashboard)

**Celery Example:**
```python
from celery import Celery

celery_app = Celery('autotuner',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/1')

@celery_app.task
def run_tuning_task(task_id: str):
    orch = AutotunerOrchestrator(...)
    result = orch.run_task(task_file)
    return result

# FastAPI integration
@app.post("/api/tasks/{task_id}/start")
def start_task(task_id: str):
    task = run_tuning_task.delay(task_id)
    return {"task_id": task.id, "status": "queued"}
```

**When to Use:**
- Production deployment
- > 10 concurrent tasks
- Need advanced features (scheduling, monitoring)
- Multi-broker setup required

**Comparison:**
| Feature | ARQ | Celery | RQ |
|---------|-----|--------|-----|
| Async Support | ✅ Native | ⚠️ Via gevent | ❌ Sync only |
| Setup Complexity | ⭐ Simple | ⭐⭐⭐ Complex | ⭐⭐ Moderate |
| Performance | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Features | ⭐⭐ Basic | ⭐⭐⭐ Advanced | ⭐⭐ Moderate |
| FastAPI Fit | ✅ Perfect | ⚠️ OK | ⚠️ OK |

**Recommendation**: Start with **ARQ** for MVP, migrate to **Celery** if needed for production

---

### Database: **PostgreSQL** (Production) or **SQLite** (MVP) ✅

#### Option A: **SQLite** (Recommended for MVP)

**Why SQLite?**
- ✅ **Zero Configuration**: File-based, no server needed
- ✅ **Fast Development**: Instant setup, easy testing
- ✅ **Embedded**: No separate process
- ✅ **SQLAlchemy Support**: Easy migration to PostgreSQL later
- ✅ **Sufficient for Single-User**: < 1000 concurrent writes/sec

**SQLite Example:**
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

# Async SQLite (with aiosqlite)
engine = create_async_engine("sqlite+aiosqlite:///./autotuner.db")
```

**When to Use:**
- MVP/prototype development
- Single-user deployment
- < 100 tasks/day
- Development/testing environments

#### Option B: **PostgreSQL** (Recommended for Production)

**Why PostgreSQL?**
- ✅ **Production-Ready**: Industry standard for web apps
- ✅ **ACID Compliance**: Data integrity guarantees
- ✅ **Concurrent Writes**: Handle multiple users writing simultaneously
- ✅ **JSON Support**: Native JSONB for storing metrics
- ✅ **Full-Text Search**: Built-in search capabilities
- ✅ **Scalability**: Handles millions of rows efficiently

**PostgreSQL Example:**
```python
# Async PostgreSQL (with asyncpg)
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/autotuner"
)
```

**When to Use:**
- Production deployment
- Multi-user environment
- > 100 tasks/day
- Need concurrent access
- Advanced querying required

**Migration Path:**
- Start with SQLite + SQLAlchemy ORM
- Switch connection string to PostgreSQL when scaling
- Minimal code changes required

---

### Real-Time Updates: **Server-Sent Events (SSE)** ✅

**Why SSE over WebSockets?**
- ✅ **Simpler Protocol**: HTTP-based, easier to implement
- ✅ **Unidirectional**: Perfect for progress updates (server → client only)
- ✅ **Auto Reconnection**: Built-in connection recovery
- ✅ **Firewall-Friendly**: Works through proxies/firewalls
- ✅ **FastAPI Support**: Built-in with StreamingResponse

**SSE Example (FastAPI):**
```python
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/api/tasks/{task_id}/progress")
async def task_progress(task_id: str):
    async def event_stream():
        while True:
            # Get progress from task queue/database
            progress = await get_task_progress(task_id)

            yield f"data: {json.dumps(progress)}\\n\\n"

            if progress['status'] in ['completed', 'failed']:
                break

            await asyncio.sleep(1)  # Update every second

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
```

**Frontend (JavaScript):**
```javascript
const eventSource = new EventSource('/api/tasks/123/progress');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateProgressBar(data.progress);
};
```

**SSE vs WebSocket Comparison:**
| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client | Bidirectional |
| Protocol | HTTP | WS/WSS |
| Complexity | ⭐ Simple | ⭐⭐ Moderate |
| Reconnection | ✅ Automatic | ⚠️ Manual |
| Firewall | ✅ Friendly | ⚠️ May block |
| Use Case | Updates, notifications | Chat, gaming |

**When to Use SSE**: Progress updates, notifications, live dashboards
**When to Use WebSocket**: Chat, collaborative editing, gaming

**Recommendation**: Use **SSE** for autotuner progress updates (simpler, sufficient)

---

### Frontend Framework: **React + TypeScript** or **Vue 3 + TypeScript** ✅

#### Option A: **React + TypeScript** (Recommended)

**Why React?**
- ✅ **Most Popular**: 220K+ GitHub stars, largest ecosystem
- ✅ **Job Market**: Most in-demand frontend skill
- ✅ **Component Libraries**: Material-UI, Ant Design, Chakra UI
- ✅ **Data Visualization**: Recharts, Victory, Plotly
- ✅ **TypeScript Support**: Excellent type safety
- ✅ **React Query**: Perfect for API data fetching/caching

**React Stack:**
```typescript
// Tech Stack
- React 18 (with Hooks)
- TypeScript 5
- Vite (build tool)
- TanStack Query (React Query) - API state management
- Zustand or Jotai - Client state management
- React Router - Navigation
- Recharts - Data visualization
- Tailwind CSS or MUI - Styling
```

**Example Component:**
```typescript
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis } from 'recharts';

function TaskResults({ taskId }: { taskId: string }) {
    const { data, isLoading } = useQuery({
        queryKey: ['task', taskId],
        queryFn: () => fetch(`/api/tasks/${taskId}`).then(r => r.json()),
    });

    if (isLoading) return <div>Loading...</div>;

    return (
        <LineChart data={data.metrics}>
            <Line dataKey="latency" stroke="#8884d8" />
        </LineChart>
    );
}
```

#### Option B: **Vue 3 + TypeScript** (Alternative)

**Why Vue?**
- ✅ **Easier Learning Curve**: Simpler than React for beginners
- ✅ **Great Documentation**: Comprehensive, well-organized
- ✅ **Composition API**: Modern, TypeScript-friendly
- ✅ **Performance**: Slightly faster than React
- ✅ **All-in-One**: Router, state management included

**Vue Stack:**
```typescript
// Tech Stack
- Vue 3 (Composition API)
- TypeScript 5
- Vite
- Pinia - State management
- Vue Router - Navigation
- Chart.js with vue-chartjs
- Naive UI or Element Plus - Component library
```

**Framework Comparison:**
| Feature | React | Vue 3 | Svelte |
|---------|-------|-------|--------|
| Popularity | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Learning Curve | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Ecosystem | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Performance | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| TypeScript | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Job Market | ⭐⭐⭐ | ⭐⭐ | ⭐ |

**Recommendation**:
- **React** for larger team, job market, ecosystem
- **Vue** for faster development, simpler learning curve

---

## 📦 Complete Recommended Stack

### 🥇 **Option 1: Modern Async Stack (Recommended for 2025)**

```yaml
Backend:
  Framework: FastAPI 0.104+
  Task Queue: ARQ (MVP) → Celery (Production)
  Database: SQLite (MVP) → PostgreSQL (Production)
  ORM: SQLAlchemy 2.0 (async)
  Real-time: Server-Sent Events (SSE)
  Cache: Redis 7+

Frontend:
  Framework: React 18 + TypeScript 5
  Build Tool: Vite 5
  State: TanStack Query + Zustand
  UI Library: Material-UI or Tailwind CSS
  Charts: Recharts or Plotly

DevOps:
  Container: Docker + Docker Compose
  Reverse Proxy: Nginx or Traefik
  Monitoring: Prometheus + Grafana (optional)

Development:
  Code Quality: black-with-tabs, ESLint, Prettier
  Testing: pytest, React Testing Library
  API Docs: Auto-generated (FastAPI Swagger)
```

**Pros:**
- ✅ Modern, best-practice stack for 2025
- ✅ Excellent async performance
- ✅ Great developer experience
- ✅ Easy to scale from MVP to production
- ✅ Strong TypeScript support throughout

**Cons:**
- ⚠️ ARQ is less battle-tested than Celery
- ⚠️ Smaller community than Django/Flask

---

### 🥈 **Option 2: Production-Ready Stack (Conservative)**

```yaml
Backend:
  Framework: FastAPI 0.104+
  Task Queue: Celery 5.3+
  Database: PostgreSQL 16+
  ORM: SQLAlchemy 2.0 (async)
  Real-time: Server-Sent Events (SSE)
  Cache: Redis 7+
  Message Broker: RabbitMQ or Redis

Frontend:
  Framework: Vue 3 + TypeScript 5
  Build Tool: Vite 5
  State: Pinia
  UI Library: Naive UI or Element Plus
  Charts: Chart.js
```

**Pros:**
- ✅ Battle-tested components (Celery, PostgreSQL)
- ✅ Easier Vue learning curve
- ✅ Ready for production from day 1
- ✅ Better monitoring (Flower for Celery)

**Cons:**
- ⚠️ More complex setup
- ⚠️ Higher infrastructure requirements
- ⚠️ Slightly slower development

---

## 🚀 Migration Path (MVP → Production)

### Phase 1: MVP (Week 1-2)
```
FastAPI + ARQ + SQLite + React
- Single-user deployment
- Local development
- File-based database
- Simple task queue
```

### Phase 2: Multi-User (Week 3-4)
```
FastAPI + ARQ + PostgreSQL + React
- Add authentication (JWT)
- Switch to PostgreSQL
- User management
- Task ownership
```

### Phase 3: Production (Month 2)
```
FastAPI + Celery + PostgreSQL + React + Redis
- Migrate to Celery for advanced features
- Add monitoring (Flower, Grafana)
- Horizontal scaling
- Load balancing (Nginx)
```

---

## 🔧 Implementation Priorities

### Week 1: Backend Foundation
1. FastAPI project structure
2. Database models (SQLAlchemy)
3. CRUD endpoints for tasks
4. ARQ task queue integration
5. Orchestrator integration

### Week 2: Real-Time & Frontend
1. SSE endpoints for progress
2. React project setup (Vite)
3. Task submission form
4. Task list/dashboard
5. Basic result display

### Week 3: Visualization & Polish
1. Chart integration (Recharts)
2. Result comparison views
3. Error handling
4. Loading states
5. Responsive design

### Week 4: Testing & Deployment
1. Unit tests (pytest)
2. Integration tests
3. Docker Compose setup
4. CI/CD pipeline
5. Documentation

---

## 📊 Technology Justification Matrix

| Requirement | Technology | Justification | Alternatives |
|-------------|-----------|---------------|--------------|
| API Framework | FastAPI | Best async, auto docs, modern | Flask, Django |
| Task Queue | ARQ/Celery | Async support, reliability | RQ, Dramatiq |
| Database | PostgreSQL | Production-ready, JSONB | SQLite, MySQL |
| Real-time | SSE | Simple, unidirectional | WebSocket |
| Frontend | React+TS | Ecosystem, job market | Vue, Svelte |
| State Mgmt | TanStack Query | Server state caching | Redux, Zustand |
| Styling | Tailwind/MUI | Rapid development | Custom CSS |
| Charts | Recharts | React-native, composable | Chart.js, Plotly |

---

## 🎯 Success Criteria

**Performance:**
- API response time < 100ms
- Handle 10+ concurrent tasks
- Real-time updates < 1s latency

**Developer Experience:**
- Type safety throughout stack
- Auto-generated API docs
- Hot reload in development
- < 5 min setup time

**User Experience:**
- Intuitive task submission
- Clear progress indication
- Responsive on mobile
- < 2s page load time

---

## 📚 Learning Resources

**FastAPI:**
- Official Docs: https://fastapi.tiangolo.com/
- Tutorial: "FastAPI from Zero to Hero"

**React + TypeScript:**
- Official Docs: https://react.dev/
- TypeScript Handbook: https://www.typescriptlang.org/

**ARQ:**
- Docs: https://arq-docs.helpmanual.io/
- FastAPI Integration: https://github.com/tobymao/arq-fastapi

**PostgreSQL:**
- Official Docs: https://www.postgresql.org/docs/
- SQLAlchemy 2.0: https://docs.sqlalchemy.org/

---

## 🎬 Conclusion

**Recommended Stack for LLM Inference Autotuner:**

✅ **Backend**: FastAPI + ARQ + PostgreSQL + SQLAlchemy
✅ **Frontend**: React + TypeScript + Vite + TanStack Query
✅ **Real-time**: Server-Sent Events (SSE)
✅ **Infrastructure**: Docker + Redis + Nginx

**Why This Stack?**
1. **Modern & Performant**: Best-practice 2025 technologies
2. **Async-Native**: Perfect for long-running tasks
3. **Type-Safe**: TypeScript + Pydantic throughout
4. **Scalable**: Easy migration path from MVP to production
5. **Developer-Friendly**: Great DX, auto docs, fast iteration

**Next Steps:**
1. Create FastAPI project structure
2. Implement core API endpoints
3. Set up React frontend with Vite
4. Integrate ARQ for background tasks
5. Add SSE for real-time progress

**Estimated Timeline**: 2-4 weeks for functional MVP
