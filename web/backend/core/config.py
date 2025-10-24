"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
	"""Application settings."""

	# Application
	app_name: str = "LLM Inference Autotuner API"
	app_version: str = "0.1.0"
	debug: bool = True

	# Database
	database_url: str = "sqlite+aiosqlite:///./data/autotuner.db"

	# Redis (for ARQ)
	redis_host: str = "localhost"
	redis_port: int = 6379
	redis_db: int = 0

	# CORS
	cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]

	# Autotuner
	docker_model_path: str = "/mnt/data/models"
	deployment_mode: str = "docker"

	class Config:
		env_file = ".env"
		case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
	"""Get cached settings instance."""
	return Settings()
