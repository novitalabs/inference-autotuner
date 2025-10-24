"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TaskStatusEnum(str, Enum):
	"""Task status enum."""

	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class ExperimentStatusEnum(str, Enum):
	"""Experiment status enum."""

	PENDING = "pending"
	DEPLOYING = "deploying"
	BENCHMARKING = "benchmarking"
	SUCCESS = "success"
	FAILED = "failed"


# Task schemas
class TaskCreate(BaseModel):
	"""Schema for creating a task."""

	task_name: str = Field(..., description="Unique task name")
	description: Optional[str] = Field(None, description="Task description")
	model: Dict[str, Any] = Field(..., description="Model configuration")
	base_runtime: str = Field(..., description="Runtime (sglang, vllm)")
	runtime_image_tag: Optional[str] = Field(None, description="Docker image tag")
	parameters: Dict[str, Any] = Field(..., description="Parameter grid")
	optimization: Dict[str, Any] = Field(..., description="Optimization settings")
	benchmark: Dict[str, Any] = Field(..., description="Benchmark configuration")
	deployment_mode: str = Field("docker", description="Deployment mode")


class TaskUpdate(BaseModel):
	"""Schema for updating a task."""

	description: Optional[str] = None
	status: Optional[TaskStatusEnum] = None


class TaskResponse(BaseModel):
	"""Schema for task response."""

	model_config = {"from_attributes": True}

	id: int
	task_name: str
	description: Optional[str]
	status: TaskStatusEnum
	model: Dict[str, Any] = Field(alias="model_config")
	base_runtime: str
	runtime_image_tag: Optional[str]
	parameters: Dict[str, Any]
	optimization: Dict[str, Any] = Field(alias="optimization_config")
	benchmark: Dict[str, Any] = Field(alias="benchmark_config")
	deployment_mode: str
	total_experiments: int
	successful_experiments: int
	best_experiment_id: Optional[int]
	created_at: datetime
	started_at: Optional[datetime]
	completed_at: Optional[datetime]
	elapsed_time: Optional[float]


class TaskListResponse(BaseModel):
	"""Schema for task list response."""

	model_config = {"from_attributes": True}

	id: int
	task_name: str
	description: Optional[str]
	status: TaskStatusEnum
	base_runtime: str
	total_experiments: int
	successful_experiments: int
	created_at: datetime
	elapsed_time: Optional[float]


# Experiment schemas
class ExperimentResponse(BaseModel):
	"""Schema for experiment response."""

	model_config = {"from_attributes": True}

	id: int
	task_id: int
	experiment_id: int
	parameters: Dict[str, Any]
	status: ExperimentStatusEnum
	error_message: Optional[str]
	metrics: Optional[Dict[str, Any]]
	objective_score: Optional[float]
	service_name: Optional[str]
	service_url: Optional[str]
	created_at: datetime
	started_at: Optional[datetime]
	completed_at: Optional[datetime]
	elapsed_time: Optional[float]


# Progress update schema
class ProgressUpdate(BaseModel):
	"""Schema for progress updates (SSE)."""

	task_id: int
	task_name: str
	status: TaskStatusEnum
	current_experiment: Optional[int] = None
	total_experiments: int
	successful_experiments: int
	progress_percent: float
	message: str
	experiment: Optional[ExperimentResponse] = None


# System schemas
class HealthResponse(BaseModel):
	"""Health check response."""

	status: str
	database: str = "ok"
	redis: str = "ok"


class SystemInfoResponse(BaseModel):
	"""System information response."""

	app_name: str
	version: str
	deployment_mode: str
	available_runtimes: List[str]
