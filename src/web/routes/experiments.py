"""
Experiment API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from web.db.session import get_db
from web.db.models import Experiment
from web.schemas import ExperimentResponse

router = APIRouter()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: int, db: AsyncSession = Depends(get_db)):
	"""Get experiment by ID."""
	result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
	experiment = result.scalar_one_or_none()

	if not experiment:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found")

	return experiment


@router.get("/task/{task_id}", response_model=List[ExperimentResponse])
async def list_task_experiments(task_id: int, db: AsyncSession = Depends(get_db)):
	"""List all experiments for a task."""
	result = await db.execute(
		select(Experiment).where(Experiment.task_id == task_id).order_by(Experiment.experiment_id)
	)
	experiments = result.scalars().all()

	return experiments
