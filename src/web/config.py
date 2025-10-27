"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os


def get_default_database_url() -> str:
	"""Get default database URL in user's home directory."""
	return f"sqlite+aiosqlite:///{Path.home()}/.local/share/inference-autotuner/autotuner.db"


class Settings(BaseSettings):
	"""Application settings."""

	model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

	# Application
	app_name: str = "LLM Inference Autotuner API"
	app_version: str = "0.1.0"
	debug: bool = True

	# Database
	# Store database in user's home directory by default
	# Use environment variable DATABASE_URL to override
	database_url: str = Field(default_factory=get_default_database_url)

	# Redis (for ARQ)
	redis_host: str = "localhost"
	redis_port: int = 6379
	redis_db: int = 0

	# CORS
	cors_origins: list = [
		"http://localhost:3000",
		"http://localhost:3001",
		"http://localhost:3002",
		"http://localhost:5173"
	]

	# Autotuner
	docker_model_path: str = "/mnt/data/models"
	deployment_mode: str = "docker"


@lru_cache()
def get_settings() -> Settings:
	"""Get cached settings instance."""
	return Settings()
