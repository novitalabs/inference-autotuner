#!/usr/bin/env python3
"""
Fix Task 7: Remove duplicate experiments and rebuild checkpoint from database.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, delete
from src.web.config import get_settings
from src.web.db.models import Task, Experiment, ExperimentStatus
from src.web.workers.checkpoint import TaskCheckpoint

async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        # Get Task 7
        result = await db.execute(select(Task).where(Task.id == 7))
        task = result.scalar_one_or_none()

        if not task:
            print("❌ Task 7 not found")
            return

        print(f"📋 Task: {task.task_name}")
        print(f"   Status: {task.status}")
        print()

        # Get all experiments
        result = await db.execute(
            select(Experiment)
            .where(Experiment.task_id == 7)
            .order_by(Experiment.id)
        )
        all_experiments = result.scalars().all()

        print(f"📊 Found {len(all_experiments)} experiment records")
        print()

        # Group by experiment_id to find duplicates
        by_exp_id = {}
        for exp in all_experiments:
            if exp.experiment_id not in by_exp_id:
                by_exp_id[exp.experiment_id] = []
            by_exp_id[exp.experiment_id].append(exp)

        # Print summary
        print("Experiment Summary:")
        for exp_id in sorted(by_exp_id.keys()):
            exps = by_exp_id[exp_id]
            print(f"  Exp {exp_id}: {len(exps)} records", end="")
            if len(exps) > 1:
                print(f" ⚠️  DUPLICATE")
            else:
                print()

        print()

        # Strategy: Keep the LAST record for each experiment_id (most recent)
        # Delete earlier duplicates
        experiments_to_delete = []
        experiments_to_keep = []

        for exp_id, exps in by_exp_id.items():
            if len(exps) > 1:
                # Sort by ID (earlier = lower ID)
                exps_sorted = sorted(exps, key=lambda e: e.id)
                # Keep the last one, delete the rest
                experiments_to_delete.extend(exps_sorted[:-1])
                experiments_to_keep.append(exps_sorted[-1])
                print(f"Exp {exp_id}: Keeping record #{exps_sorted[-1].id}, deleting {len(exps_sorted)-1} duplicate(s)")
            else:
                experiments_to_keep.append(exps[0])

        if experiments_to_delete:
            print()
            print(f"🗑️  Deleting {len(experiments_to_delete)} duplicate records...")
            for exp in experiments_to_delete:
                await db.delete(exp)
            await db.commit()
            print("✅ Duplicates removed")
        else:
            print("✅ No duplicates to remove")

        print()

        # Now rebuild checkpoint from remaining experiments
        experiments_to_keep.sort(key=lambda e: e.experiment_id)
        successful_exps = [e for e in experiments_to_keep if e.status == ExperimentStatus.SUCCESS]

        print(f"📝 Rebuilding checkpoint from {len(successful_exps)} successful experiments...")
        print()

        # Build history for checkpoint
        history = []
        best_score = float('inf')
        best_experiment_id = None

        for exp in successful_exps:
            print(f"  Exp {exp.experiment_id}: score={exp.objective_score:.2f}")
            history.append({
                "parameters": exp.parameters,
                "objective_score": exp.objective_score,
                "metrics": exp.metrics or {}
            })

            if exp.objective_score is not None and exp.objective_score < best_score:
                best_score = exp.objective_score
                best_experiment_id = exp.id

        print()
        print(f"🏆 Best experiment: #{best_experiment_id} with score {best_score:.2f}")
        print()

        # Create new checkpoint with history
        checkpoint_data = {
            "checkpoint": {
                "iteration": len(successful_exps),
                "best_score": best_score,
                "best_experiment_id": best_experiment_id,
                "strategy_state": {
                    "strategy_class": "BayesianStrategy",
                    "parameter_spec": task.parameters,
                    "objective": task.optimization_config.get("objective", "maximize_throughput"),
                    "trial_count": len(successful_exps),
                    "max_iterations": task.optimization_config.get("max_iterations", 50),
                    "n_initial_random": task.optimization_config.get("n_initial_random", 5),
                    "history": history  # ✅ Include history!
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }

        # Update task metadata
        task.task_metadata = checkpoint_data
        await db.commit()

        print("✅ Checkpoint rebuilt with complete history")
        print()
        print(f"📦 Checkpoint details:")
        print(f"   Iteration: {len(successful_exps)}")
        print(f"   Best score: {best_score:.2f}")
        print(f"   History entries: {len(history)}")
        print()
        print("✅ Task 7 is ready to resume from experiment {len(successful_exps) + 1}")

if __name__ == "__main__":
    asyncio.run(main())
