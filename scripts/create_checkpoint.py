#!/usr/bin/env python3
"""
Create checkpoint for Task 7 to resume from iteration 2.
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select
from src.web.config import get_settings
from src.web.db.models import Task, Experiment
from src.web.workers.checkpoint import TaskCheckpoint
from src.utils.optimizer import create_optimization_strategy

async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        # Get Task 7
        result = await db.execute(select(Task).where(Task.id == 7))
        task = result.scalar_one_or_none()

        if not task:
            print("Task 7 not found")
            return

        # Get completed experiments
        result = await db.execute(
            select(Experiment)
            .where(Experiment.task_id == 7)
            .where(Experiment.status == "SUCCESS")
            .order_by(Experiment.id)
        )
        experiments = result.scalars().all()

        print(f"Found {len(experiments)} completed experiments")
        for exp in experiments:
            print(f"  Experiment {exp.experiment_id}: score={exp.objective_score}, params={exp.parameters}")

        # Create strategy and replay experiments
        optimization_config = task.optimization_config or {}
        strategy = create_optimization_strategy(optimization_config, task.parameters)

        # Tell strategy about completed experiments
        for exp in experiments:
            if exp.objective_score is not None:
                # For Bayesian strategy, we need to suggest first, then tell
                # But we already know the parameters, so we need to replay differently
                suggested_params = strategy.suggest_parameters()
                print(f"  Suggested params: {suggested_params}")

                # Now tell the strategy about the actual result
                strategy.tell_result(
                    parameters=exp.parameters,
                    objective_score=exp.objective_score,
                    metrics=exp.metrics or {}
                )
                print(f"  Replayed experiment {exp.experiment_id} to strategy")

        # Find best experiment
        best_experiment = min(experiments, key=lambda e: e.objective_score if e.objective_score is not None else float('inf'))
        best_score = best_experiment.objective_score
        best_experiment_id = best_experiment.id
        iteration = len(experiments)

        print(f"\nBest experiment: {best_experiment.experiment_id} (id={best_experiment_id}, score={best_score})")
        print(f"Iteration: {iteration}")

        # Create checkpoint
        updated_metadata = TaskCheckpoint.save_checkpoint(
            task_metadata=task.metadata or {},
            iteration=iteration,
            best_score=best_score,
            best_experiment_id=best_experiment_id,
            strategy_state=strategy.get_state(),
        )

        task.metadata = updated_metadata
        await db.commit()

        print(f"\nCheckpoint saved for Task 7")
        print(f"  Iteration: {iteration}")
        print(f"  Best score: {best_score}")
        print(f"  Best experiment ID: {best_experiment_id}")

if __name__ == "__main__":
    asyncio.run(main())
