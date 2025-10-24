"""
Task management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from db.session import get_db
from db.models import Task, TaskStatus
from schemas import TaskCreate, TaskUpdate, TaskResponse, TaskListResponse

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
	from workers import enqueue_autotuning_task

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
