"""
ARQ worker configuration and task functions.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
# workers/autotuner_worker.py -> web -> src -> project_root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update
from datetime import datetime
from typing import Dict, Any
import io

from src.web.config import get_settings
from src.web.db.models import Task, Experiment, TaskStatus, ExperimentStatus
from src.orchestrator import AutotunerOrchestrator
from src.utils.optimizer import generate_parameter_grid, calculate_objective_score

settings = get_settings()

# Create database session for workers
engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class StreamToLogger:
	"""File-like stream object that redirects writes to a logger instance."""

	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ""

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass


def setup_task_logging(task_id: int):
	"""Setup logging for a specific task.

	Args:
	    task_id: Task ID

	Returns:
	    Logger instance configured for this task
	"""
	# Create log directory
	log_dir = Path.home() / ".local/share/inference-autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f"task_{task_id}.log"

	# Create logger for this task
	logger = logging.getLogger(f"task_{task_id}")
	logger.setLevel(logging.DEBUG)
	logger.handlers.clear()  # Remove any existing handlers
	logger.propagate = False  # CRITICAL: Don't propagate to parent loggers to avoid recursion

	# Create file handler
	file_handler = logging.FileHandler(log_file, mode="a")
	file_handler.setLevel(logging.DEBUG)

	# Create console handler - IMPORTANT: Use sys.__stdout__ (the true original)
	# sys.__stdout__ is saved by Python at startup and is never redirected
	console_handler = logging.StreamHandler(sys.__stdout__)
	console_handler.setLevel(logging.INFO)

	# Create formatter
	formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	# Add handlers to logger
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# Redirect stdout and stderr to logger
	sys.stdout = StreamToLogger(logger, logging.INFO)
	sys.stderr = StreamToLogger(logger, logging.ERROR)

	return logger


async def run_autotuning_task(ctx: Dict[str, Any], task_id: int) -> Dict[str, Any]:
	"""Run autotuning task in background.

	Args:
	    ctx: ARQ context
	    task_id: Database task ID

	Returns:
	    Task summary dict
	"""
	# Setup logging for this task
	logger = setup_task_logging(task_id)

	async with AsyncSessionLocal() as db:
		# Get task from database
		result = await db.execute(select(Task).where(Task.id == task_id))
		task = result.scalar_one_or_none()

		if not task:
			error_msg = f"Task {task_id} not found"
			logger.error(error_msg)
			return {"error": error_msg}

		try:
			logger.info(f"[ARQ Worker] Starting task: {task.task_name}")

			# Update task status
			task.status = TaskStatus.RUNNING
			task.started_at = datetime.utcnow()
			await db.commit()

			# Create task configuration dict (similar to JSON task file)
			task_config = {
				"task_name": task.task_name,
				"description": task.description or "",
				"model": task.model_config,
				"base_runtime": task.base_runtime,
				"runtime_image_tag": task.runtime_image_tag,
				"parameters": task.parameters,
				"optimization": task.optimization_config,
				"benchmark": task.benchmark_config,
			}

			# Generate parameter grid
			param_grid = generate_parameter_grid(task.parameters)
			total_experiments = len(param_grid)
			task.total_experiments = total_experiments
			await db.commit()

			logger.info(f"[ARQ Worker] Generated {total_experiments} parameter combinations")

			# Create orchestrator
			orchestrator = AutotunerOrchestrator(
				deployment_mode=task.deployment_mode,
				use_direct_benchmark=True,
				docker_model_path=settings.docker_model_path,
				verbose=False,
				http_proxy=settings.http_proxy,
				https_proxy=settings.https_proxy,
				no_proxy=settings.no_proxy,
				hf_token=settings.hf_token,
			)

			# Run experiments
			best_score = float("inf")
			best_experiment_id = None

			for idx, params in enumerate(param_grid, 1):
				logger.info(f"[ARQ Worker] Running experiment {idx}/{total_experiments} with params: {params}")

				# Create experiment record
				db_experiment = Experiment(
					task_id=task_id,
					experiment_id=idx,
					parameters=params,
					status=ExperimentStatus.PENDING,
				)
				db.add(db_experiment)
				await db.commit()
				await db.refresh(db_experiment)

				# Update status to deploying
				db_experiment.status = ExperimentStatus.DEPLOYING
				db_experiment.started_at = datetime.utcnow()
				await db.commit()

				logger.info(f"[Experiment {idx}] Status: DEPLOYING")

				# Run experiment using orchestrator
				try:
					result = orchestrator.run_experiment(task_config, idx, params)

					logger.info(f"[Experiment {idx}] Status: {result['status'].upper()}")
					if result.get("metrics"):
						logger.info(f"[Experiment {idx}] Metrics: {result['metrics']}")

					# Save container logs if available (Docker mode)
					if result.get("container_logs"):
						logger.info(f"[Experiment {idx}] ========== Container Logs ==========")
						for line in result["container_logs"].splitlines():
							logger.info(f"[Experiment {idx}] {line}")
						logger.info(f"[Experiment {idx}] ========== End Container Logs ==========")

					# Update experiment with results
					db_experiment.status = (
						ExperimentStatus.SUCCESS if result["status"] == "success" else ExperimentStatus.FAILED
					)
					db_experiment.metrics = result.get("metrics")
					db_experiment.objective_score = result.get("objective_score")
					db_experiment.completed_at = datetime.utcnow()

					if db_experiment.started_at:
						elapsed = (db_experiment.completed_at - db_experiment.started_at).total_seconds()
						db_experiment.elapsed_time = elapsed
						logger.info(f"[Experiment {idx}] Completed in {elapsed:.2f}s")

					# Check if this is the best experiment
					if result["status"] == "success" and result.get("objective_score") is not None:
						task.successful_experiments += 1
						if result["objective_score"] < best_score:
							best_score = result["objective_score"]
							best_experiment_id = db_experiment.id
							logger.info(f"[Experiment {idx}] New best score: {best_score:.4f}")

					await db.commit()

				except Exception as e:
					logger.error(f"[Experiment {idx}] Failed: {e}", exc_info=True)
					db_experiment.status = ExperimentStatus.FAILED
					db_experiment.error_message = str(e)
					await db.commit()

			# Update task with final results
			# Refresh task object to ensure it's properly tracked by the session
			await db.refresh(task)
			task.status = TaskStatus.COMPLETED
			task.completed_at = datetime.utcnow()
			task.best_experiment_id = best_experiment_id

			if task.started_at:
				elapsed = (task.completed_at - task.started_at).total_seconds()
				task.elapsed_time = elapsed
				logger.info(f"[ARQ Worker] Task completed in {elapsed:.2f}s - Best experiment: {best_experiment_id}")

			await db.commit()
			await db.refresh(task)  # Ensure changes are reflected

			logger.info(
				f"[ARQ Worker] Task finished: {task.task_name} - {task.successful_experiments}/{total_experiments} successful"
			)
			return {"task_id": task_id, "task_name": task.task_name, "status": "completed"}

		except Exception as e:
			logger.error(f"[ARQ Worker] Task failed: {e}", exc_info=True)
			task.status = TaskStatus.FAILED
			task.completed_at = datetime.utcnow()
			await db.commit()
			return {"task_id": task_id, "error": str(e)}
		finally:
			# Restore stdout and stderr
			sys.stdout = sys.__stdout__
			sys.stderr = sys.__stderr__
			# Remove handlers to prevent memory leaks
			logger.handlers.clear()


# ARQ worker settings
class WorkerSettings:
	"""ARQ worker configuration."""

	redis_settings = RedisSettings(
		host=settings.redis_host,
		port=settings.redis_port,
		database=settings.redis_db,
	)

	functions = [run_autotuning_task]

	# Worker config
	max_jobs = 5  # Maximum concurrent jobs
	job_timeout = 7200  # 2 hours timeout per job
	keep_result = 3600  # Keep results for 1 hour
