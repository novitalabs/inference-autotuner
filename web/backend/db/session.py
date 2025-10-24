"""
Database session management.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from core.config import get_settings
from db.models import Base

settings = get_settings()

# Create async engine
engine = create_async_engine(
	settings.database_url,
	echo=settings.debug,
	future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
	engine,
	class_=AsyncSession,
	expire_on_commit=False,
)


async def init_db():
	"""Initialize database (create tables)."""
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
	"""Dependency to get database session."""
	async with AsyncSessionLocal() as session:
		try:
			yield session
		finally:
			await session.close()
