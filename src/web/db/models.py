"""
Database models for tasks, experiments, and parameter presets.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, ForeignKey, Enum as SQLEnum, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class TaskStatus(str, enum.Enum):
	"""Task status enum."""

	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class Task(Base):
	"""Autotuning task model."""

	__tablename__ = "tasks"

	id = Column(Integer, primary_key=True, index=True)
	task_name = Column(String, unique=True, index=True, nullable=False)
	description = Column(String)
	status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)

	# Configuration
	model_config = Column(JSON, nullable=False)  # model name, namespace
	base_runtime = Column(String, nullable=False)  # sglang, vllm
	runtime_image_tag = Column(String)
	parameters = Column(JSON, nullable=False)  # parameter grid
	optimization_config = Column(JSON, nullable=False)  # strategy, objective
	benchmark_config = Column(JSON, nullable=False)  # benchmark settings
	slo_config = Column(JSON, nullable=True)  # SLO constraints (optional)
	config_metadata = Column("metadata", JSON, nullable=True)  # Additional metadata (e.g., applied_layers)

	# Deployment mode
	deployment_mode = Column(String, default="docker")  # docker, ome

	# Results
	total_experiments = Column(Integer, default=0)
	successful_experiments = Column(Integer, default=0)
	best_experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

	# Timing
	created_at = Column(DateTime, default=datetime.utcnow)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	elapsed_time = Column(Float, nullable=True)

	# Relationships
	experiments = relationship("Experiment", back_populates="task", foreign_keys="Experiment.task_id")
	best_experiment = relationship("Experiment", foreign_keys=[best_experiment_id], post_update=True)


class ExperimentStatus(str, enum.Enum):
	"""Experiment status enum."""

	PENDING = "pending"
	DEPLOYING = "deploying"
	BENCHMARKING = "benchmarking"
	SUCCESS = "success"
	FAILED = "failed"


class Experiment(Base):
	"""Individual experiment (single parameter configuration) model."""

	__tablename__ = "experiments"

	id = Column(Integer, primary_key=True, index=True)
	task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
	experiment_id = Column(Integer, nullable=False)  # Sequential ID within task

	# Configuration
	parameters = Column(JSON, nullable=False)

	# Status
	status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING)
	error_message = Column(String, nullable=True)

	# Results
	metrics = Column(JSON, nullable=True)
	objective_score = Column(Float, nullable=True, index=True)

	# Service info
	service_name = Column(String, nullable=True)
	service_url = Column(String, nullable=True)

	# Timing
	created_at = Column(DateTime, default=datetime.utcnow)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	elapsed_time = Column(Float, nullable=True)

	# Relationship
	task = relationship("Task", back_populates="experiments", foreign_keys=[task_id])


class ParameterPreset(Base):
	"""Parameter preset model for reusable parameter configurations."""

	__tablename__ = "parameter_presets"

	id = Column(Integer, primary_key=True, autoincrement=True)
	name = Column(String(255), nullable=False, unique=True, index=True)
	description = Column(Text)
	category = Column(String(100), index=True)
	runtime = Column(String(50), index=True)  # Runtime: sglang, vllm, or None for universal
	is_system = Column(Boolean, default=False, index=True)
	parameters = Column(JSON, nullable=False)
	preset_metadata = Column("metadata", JSON)  # Use different Python name
	created_at = Column(DateTime(timezone=True), server_default=func.now())
	updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

	def to_dict(self):
		"""Convert model to dictionary."""
		return {
			"id": self.id,
			"name": self.name,
			"description": self.description,
			"category": self.category,
			"runtime": self.runtime,
			"is_system": self.is_system,
			"parameters": self.parameters,
			"metadata": self.preset_metadata,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}
