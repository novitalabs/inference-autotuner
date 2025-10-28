"""
Task management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
import asyncio
from pathlib import Path
import os

from web.db.session import get_db
from web.db.models import Task, TaskStatus
from web.schemas import TaskCreate, TaskUpdate, TaskResponse, TaskListResponse

router = APIRouter()


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(task_data: TaskCreate, db: AsyncSession = Depends(get_db)):
	"""Create a new autotuning task."""
	# Check if task name already exists
	result = await db.execute(select(Task).where(Task.task_name == task_data.task_name))
	existing_task = result.scalar_one_or_none()

	if existing_task:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task '{task_data.task_name}' already exists"
		)

	# Create new task
	db_task = Task(
		task_name=task_data.task_name,
		description=task_data.description,
		model_config=task_data.model,
		base_runtime=task_data.base_runtime,
		runtime_image_tag=task_data.runtime_image_tag,
		parameters=task_data.parameters,
		optimization_config=task_data.optimization,
		benchmark_config=task_data.benchmark,
		deployment_mode=task_data.deployment_mode,
		status=TaskStatus.PENDING,
	)

	db.add(db_task)
	await db.commit()
	await db.refresh(db_task)

	return db_task


@router.get("/", response_model=List[TaskListResponse])
async def list_tasks(skip: int = 0, limit: int = 100, status_filter: str = None, db: AsyncSession = Depends(get_db)):
	"""List all autotuning tasks."""
	query = select(Task).order_by(Task.created_at.desc())

	if status_filter:
		query = query.where(Task.status == status_filter)

	query = query.offset(skip).limit(limit)
	result = await db.execute(query)
	tasks = result.scalars().all()

	return tasks


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Get task by ID."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	return task


@router.get("/name/{task_name}", response_model=TaskResponse)
async def get_task_by_name(task_name: str, db: AsyncSession = Depends(get_db)):
	"""Get task by name."""
	result = await db.execute(select(Task).where(Task.task_name == task_name))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_name}' not found")

	return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, task_update: TaskUpdate, db: AsyncSession = Depends(get_db)):
	"""Update task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Update fields
	if task_update.description is not None:
		task.description = task_update.description
	if task_update.status is not None:
		task.status = task_update.status

	await db.commit()
	await db.refresh(task)

	return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Delete task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Don't allow deletion of running tasks
	if task.status == TaskStatus.RUNNING:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete running task. Cancel it first."
		)

	await db.delete(task)
	await db.commit()


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Start autotuning task execution."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	if task.status != TaskStatus.PENDING:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task must be in PENDING status to start")

	# Update status to RUNNING
	task.status = TaskStatus.RUNNING
	from datetime import datetime

	task.started_at = datetime.utcnow()

	await db.commit()
	await db.refresh(task)

	# Enqueue ARQ job
	from web.workers import enqueue_autotuning_task

	job_id = await enqueue_autotuning_task(task.id)
	print(f"[API] Enqueued task {task.id} with job_id: {job_id}")

	return task


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Cancel running task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	if task.status != TaskStatus.RUNNING:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task is not running")

	task.status = TaskStatus.CANCELLED
	await db.commit()
	await db.refresh(task)

	# TODO: Cancel ARQ job (will be implemented in ARQ setup)

	return task


@router.post("/{task_id}/restart", response_model=TaskResponse)
async def restart_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Restart a completed, failed, or cancelled task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Only allow restart for completed, failed, or cancelled tasks
	if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=f"Task must be completed, failed, or cancelled to restart. Current status: {task.status}"
		)

	# Reset task to PENDING status
	task.status = TaskStatus.PENDING
	from datetime import datetime
	task.started_at = None
	task.completed_at = None
	task.elapsed_time = None
	# Reset experiment counters
	task.successful_experiments = 0
	task.best_experiment_id = None

	await db.commit()
	await db.refresh(task)

	return task


def get_task_log_file(task_id: int) -> Path:
	"""Get the log file path for a task."""
	log_dir = Path.home() / ".local/share/inference-autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	return log_dir / f"task_{task_id}.log"


async def stream_log_file(log_file: Path, follow: bool = False):
	"""Stream log file contents, optionally following new lines."""
	try:
		# If file doesn't exist yet, wait for it (up to 30 seconds)
		if not log_file.exists():
			if follow:
				yield "data: Waiting for log file to be created...\n\n"
				for _ in range(30):
					await asyncio.sleep(1)
					if log_file.exists():
						break
				else:
					yield "data: Log file not found. Task may not have started yet.\n\n"
					return
			else:
				yield "data: Log file not found.\n\n"
				return

		# Stream existing content
		with open(log_file, "r") as f:
			for line in f:
				yield f"data: {line.rstrip()}\n\n"

		# If follow mode, watch for new lines
		if follow:
			last_pos = log_file.stat().st_size
			while True:
				await asyncio.sleep(0.5)  # Poll every 500ms
				
				# Check if file still exists
				if not log_file.exists():
					yield "data: [Log file removed]\n\n"
					break
					
				current_size = log_file.stat().st_size
				if current_size > last_pos:
					with open(log_file, "r") as f:
						f.seek(last_pos)
						for line in f:
							yield f"data: {line.rstrip()}\n\n"
					last_pos = current_size
				elif current_size < last_pos:
					# File was truncated, start from beginning
					last_pos = 0
					yield "data: [Log file was truncated]\n\n"
	except Exception as e:
		yield f"data: Error reading log: {str(e)}\n\n"


@router.get("/{task_id}/logs")
async def get_task_logs(
	task_id: int,
	follow: bool = False,
	db: AsyncSession = Depends(get_db)
):
	"""
	Get task execution logs.
	
	Args:
		task_id: Task ID
		follow: If True, streams logs in real-time (Server-Sent Events)
	
	Returns:
		Log content as text or streaming response
	"""
	# Verify task exists
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()
	
	if not task:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Task {task_id} not found"
		)
	
	log_file = get_task_log_file(task_id)
	
	# If follow mode, return streaming response (Server-Sent Events)
	if follow:
		return StreamingResponse(
			stream_log_file(log_file, follow=True),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no"  # Disable nginx buffering
			}
		)
	
	# Otherwise return static log content
	if not log_file.exists():
		return {"logs": "No logs available yet."}
	
	with open(log_file, "r") as f:
		logs = f.read()
	
	return {"logs": logs}


@router.delete("/{task_id}/logs", status_code=status.HTTP_204_NO_CONTENT)
async def clear_task_logs(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Clear task logs."""
	# Verify task exists
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()
	
	if not task:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Task {task_id} not found"
		)
	
	log_file = get_task_log_file(task_id)
	if log_file.exists():
		log_file.unlink()
