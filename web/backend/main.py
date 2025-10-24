"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import get_settings
from db.session import init_db
from api import tasks, experiments, system


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan events."""
	# Startup
	print("🚀 Starting LLM Inference Autotuner API...")
	await init_db()
	print("✅ Database initialized")
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
