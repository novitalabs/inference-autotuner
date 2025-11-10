#!/usr/bin/env python3
"""
Comprehensive analysis and fix for Task 7 duplicate experiments.

This script:
1. Analyzes all experiment records to understand the duplication pattern
2. Identifies which records to keep (most recent per experiment_id)
3. Removes duplicate records
4. Rebuilds checkpoint with complete history
5. Updates task status to reflect actual progress
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from web.db.models import Task, Experiment, ExperimentStatus
from web.config import get_settings


async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        # Get Task 7
        result = await db.execute(select(Task).where(Task.id == 7))
        task = result.scalar_one()

        print("=" * 80)
        print("TASK 7 ANALYSIS")
        print("=" * 80)
        print(f"Task: {task.task_name}")
        print(f"Status: {task.status}")
        print(f"Started: {task.started_at}")
        print(f"Completed: {task.completed_at}")
        print()

        # Get all experiments
        result = await db.execute(
            select(Experiment)
            .where(Experiment.task_id == 7)
            .order_by(Experiment.experiment_id, Experiment.created_at)
        )
        all_experiments = result.scalars().all()

        print(f"Total experiment records: {len(all_experiments)}")
        print()

        # Group by experiment_id
        by_exp_id = {}
        for exp in all_experiments:
            if exp.experiment_id not in by_exp_id:
                by_exp_id[exp.experiment_id] = []
            by_exp_id[exp.experiment_id].append(exp)

        print(f"Unique experiment IDs: {len(by_exp_id)}")
        print()

        # Find duplicates
        duplicates = {exp_id: exps for exp_id, exps in by_exp_id.items() if len(exps) > 1}

        if duplicates:
            print(f"Found {len(duplicates)} experiment IDs with duplicates:")
            total_duplicates = 0
            for exp_id in sorted(duplicates.keys()):
                exps = duplicates[exp_id]
                print(f"\n  Experiment {exp_id}: {len(exps)} records")
                for i, exp in enumerate(exps, 1):
                    score = f"{exp.objective_score:.2f}" if exp.objective_score else "N/A"
                    print(f"    #{i}: ID={exp.id}, Status={exp.status.value}, Score={score}, Created={exp.created_at}")
                total_duplicates += len(exps) - 1
            print(f"\nTotal duplicate records to remove: {total_duplicates}")
        else:
            print("No duplicates found!")

        print("\n" + "=" * 80)
        print("FIX PLAN")
        print("=" * 80)

        if not duplicates:
            print("No duplicates to fix!")
            return

        # For each duplicate set, keep the MOST RECENT successfully completed one
        records_to_keep = []
        records_to_delete = []

        for exp_id, exps in by_exp_id.items():
            if len(exps) == 1:
                records_to_keep.append(exps[0])
            else:
                # Sort by: 1) Success status first, 2) Most recent created_at
                exps_sorted = sorted(
                    exps,
                    key=lambda e: (
                        e.status == ExperimentStatus.SUCCESS,  # Success first
                        e.created_at or datetime.min.replace(tzinfo=timezone.utc)  # Most recent
                    ),
                    reverse=True
                )

                records_to_keep.append(exps_sorted[0])
                records_to_delete.extend(exps_sorted[1:])

        print(f"Will keep {len(records_to_keep)} records")
        print(f"Will delete {len(records_to_delete)} duplicate records")
        print()

        # Show which records we're deleting
        if records_to_delete:
            print("Records to DELETE:")
            for exp in sorted(records_to_delete, key=lambda e: e.experiment_id):
                score = f"{exp.objective_score:.2f}" if exp.objective_score else "N/A"
                print(f"  Exp {exp.experiment_id}: ID={exp.id}, Status={exp.status.value}, Score={score}")

        # Confirm deletion
        print("\n" + "=" * 80)
        response = input("Proceed with deletion? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return

        # Delete duplicates
        print("\nDeleting duplicate records...")
        for exp in records_to_delete:
            await db.delete(exp)
        await db.commit()
        print(f"✓ Deleted {len(records_to_delete)} records")

        # Rebuild checkpoint from remaining successful experiments
        print("\nRebuilding checkpoint...")

        # Get successful experiments sorted by ID
        successful_exps = sorted(
            [e for e in records_to_keep if e.status == ExperimentStatus.SUCCESS],
            key=lambda e: e.id
        )

        print(f"Found {len(successful_exps)} successful experiments:")

        # Build history
        history = []
        best_score = float('inf')
        best_experiment_id = None

        for exp in successful_exps:
            score_str = f"{exp.objective_score:.2f}" if exp.objective_score else "N/A"
            print(f"  Exp {exp.experiment_id}: score={score_str}")

            if exp.objective_score is not None:
                history.append({
                    "parameters": exp.parameters,
                    "objective_score": exp.objective_score,
                    "metrics": exp.metrics or {}
                })

                if exp.objective_score < best_score:
                    best_score = exp.objective_score
                    best_experiment_id = exp.id

        print(f"\nBest experiment: ID={best_experiment_id}, Score={best_score:.2f}")

        # Create new checkpoint
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
                    "history": history,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }

        # Update task
        task.task_metadata = checkpoint_data
        task.best_experiment_id = best_experiment_id
        task.completed_at = None  # Clear corrupted completed_at
        await db.commit()

        print(f"\n✓ Checkpoint rebuilt with {len(history)} experiments")
        print("✓ Task metadata updated")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Unique experiments: {len(records_to_keep)}")
        print(f"Successful experiments: {len(successful_exps)}")
        print(f"Checkpoint iteration: {len(successful_exps)}")
        print(f"Best score: {best_score:.2f}")
        print(f"Task can now resume from iteration {len(successful_exps) + 1}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
