"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from web.config import get_settings
from web.db.session import init_db, get_db
from web.db.seed_presets import seed_system_presets
from web.routes import tasks, experiments, system, docker, presets


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan events."""
	# Startup
	print("🚀 Starting LLM Inference Autotuner API...")
	await init_db()
	print("✅ Database initialized")

	# Seed system presets
	async for db in get_db():
		await seed_system_presets(db)
		break

	yield
	# Shutdown
	print("👋 Shutting down...")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
	title=settings.app_name,
	version=settings.app_version,
	description="API for automated LLM inference parameter tuning",
	lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(docker.router, prefix="/api/docker", tags=["docker"])
app.include_router(presets.router)


@app.get("/")
async def root():
	"""Root endpoint."""
	return {
		"name": settings.app_name,
		"version": settings.app_version,
		"status": "running",
		"docs": "/docs",
	}


@app.get("/health")
async def health():
	"""Health check endpoint."""
	return {"status": "healthy"}
