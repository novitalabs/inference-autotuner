"""
System and health API endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from web.db.session import get_db
from web.schemas import HealthResponse, SystemInfoResponse
from web.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
	"""Health check endpoint."""
	# Check database
	db_status = "ok"
	try:
		await db.execute(text("SELECT 1"))
	except Exception:
		db_status = "error"

	# Check Redis (TODO: implement when ARQ is set up)
	redis_status = "ok"

	return HealthResponse(status="healthy" if db_status == "ok" else "degraded", database=db_status, redis=redis_status)


@router.get("/info", response_model=SystemInfoResponse)
async def system_info():
	"""Get system information."""
	settings = get_settings()

	return SystemInfoResponse(
		app_name=settings.app_name,
		version=settings.app_version,
		deployment_mode=settings.deployment_mode,
		available_runtimes=["sglang", "vllm"],
	)
