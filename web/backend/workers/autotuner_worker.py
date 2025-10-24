"""
ARQ worker configuration and task functions.
"""

import sys
from pathlib import Path

# Add project root to path to import from src
# workers/autotuner_worker.py -> backend -> web -> project_root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update
from datetime import datetime
from typing import Dict, Any

from core.config import get_settings
from db.models import Task, Experiment, TaskStatus, ExperimentStatus
from src.orchestrator import AutotunerOrchestrator
from src.utils.optimizer import generate_parameter_grid, calculate_objective_score

settings = get_settings()

# Create database session for workers
engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def run_autotuning_task(ctx: Dict[str, Any], task_id: int) -> Dict[str, Any]:
	"""Run autotuning task in background.

	Args:
	    ctx: ARQ context
	    task_id: Database task ID

	Returns:
	    Task summary dict
	"""
	async with AsyncSessionLocal() as db:
		# Get task from database
		result = await db.execute(select(Task).where(Task.id == task_id))
		task = result.scalar_one_or_none()

		if not task:
			return {"error": f"Task {task_id} not found"}

		try:
			print(f"[ARQ Worker] Starting task: {task.task_name}")

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

			print(f"[ARQ Worker] Generated {total_experiments} parameter combinations")

			# Create orchestrator
			orchestrator = AutotunerOrchestrator(
				deployment_mode=task.deployment_mode,
				use_direct_benchmark=True,
				docker_model_path=settings.docker_model_path,
				verbose=False,
			)

			# Run experiments
			best_score = float("inf")
			best_experiment_id = None

			for idx, params in enumerate(param_grid, 1):
				print(f"[ARQ Worker] Running experiment {idx}/{total_experiments}")

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

				# Run experiment using orchestrator
				try:
					result = orchestrator.run_experiment(task_config, idx, params)

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

					# Check if this is the best experiment
					if result["status"] == "success" and result.get("objective_score") is not None:
						task.successful_experiments += 1
						if result["objective_score"] < best_score:
							best_score = result["objective_score"]
							best_experiment_id = db_experiment.id

					await db.commit()

				except Exception as e:
					print(f"[ARQ Worker] Experiment {idx} failed: {e}")
					db_experiment.status = ExperimentStatus.FAILED
					db_experiment.error_message = str(e)
					await db.commit()

			# Update task with final results
			task.status = TaskStatus.COMPLETED
			task.completed_at = datetime.utcnow()
			task.best_experiment_id = best_experiment_id

			if task.started_at:
				elapsed = (task.completed_at - task.started_at).total_seconds()
				task.elapsed_time = elapsed

			await db.commit()

			print(f"[ARQ Worker] Task completed: {task.task_name}")
			return {"task_id": task_id, "task_name": task.task_name, "status": "completed"}

		except Exception as e:
			print(f"[ARQ Worker] Task failed: {e}")
			task.status = TaskStatus.FAILED
			task.completed_at = datetime.utcnow()
			await db.commit()
			return {"task_id": task_id, "error": str(e)}


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
