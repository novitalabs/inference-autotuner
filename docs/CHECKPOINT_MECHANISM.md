# Checkpoint Mechanism for Long-Running Tasks

## Overview

The checkpoint mechanism enables long-running autotuning tasks to save progress after each experiment and resume from the last completed iteration. This solves the ARQ worker timeout issue where tasks were killed mid-execution without being able to complete.

## Problem Statement

**Before checkpointing:**
- ARQ worker has a hard timeout of 7200 seconds (2 hours)
- Tasks with many experiments (e.g., 50 iterations × 10 minutes each = 500 minutes total) exceed this timeout
- When timeout occurs, worker is killed immediately
- Task status stuck in RUNNING, last experiment stuck in DEPLOYING
- All progress lost - must restart from beginning

**Example failure (Task 7):**
- 50 planned iterations, each taking ~10 minutes
- Completed 12 experiments in 9.6 hours
- Hit ARQ 2-hour timeout
- Status remained RUNNING, Experiment 12 stuck in DEPLOYING
- User had to manually fix database

## Solution: Progress Checkpointing

After implementing checkpoints:
1. **After each experiment completes** (success or failure): Save checkpoint to `task.metadata`
2. **On task start**: Check for existing checkpoint and resume if found
3. **On task completion**: Clear checkpoint from metadata

### Checkpoint Data Structure

Stored in `task.metadata` JSON field:

```json
{
  "checkpoint": {
    "iteration": 12,
    "best_score": -2981.24,
    "best_experiment_id": 71,
    "strategy_state": {
      "strategy_class": "GridSearchStrategy",
      "current_index": 12,
      "param_grid": [...],
      "parameter_spec": {...},
      "objective": "minimize_latency",
      "history": [...]
    },
    "timestamp": "2025-11-10T03:32:50+00:00"
  }
}
```

## Implementation

### 1. TaskCheckpoint Class (`src/web/workers/checkpoint.py`)

Provides static methods for checkpoint management:

- **`save_checkpoint()`**: Save progress after each experiment
  - Parameters: task_metadata, iteration, best_score, best_experiment_id, strategy_state
  - Returns: Updated metadata dict with checkpoint

- **`load_checkpoint()`**: Load checkpoint from task metadata
  - Validates required fields: iteration, best_score, strategy_state
  - Returns: Checkpoint dict or None if invalid/missing

- **`clear_checkpoint()`**: Remove checkpoint after task completion

- **`should_resume()`**: Check if task should resume from checkpoint
  - Only resumes if status is RUNNING or PENDING with valid checkpoint

### 2. Strategy State Serialization (`src/utils/optimizer.py`)

All optimization strategies now support state serialization:

**OptimizationStrategy (base class):**
- `get_state()`: Serialize strategy state to dict
- `from_state(state)`: Restore strategy from serialized state (classmethod)

**GridSearchStrategy:**
- Saves: current_index, param_grid, history
- Restores: Recreates strategy and resumes from current_index

**BayesianStrategy:**
- Saves: trial_count, max_iterations, n_initial_random, history
- Restores: Recreates Optuna study and re-populates with history trials

**RandomSearchStrategy:**
- Saves: trial_count, max_iterations, history
- Restores: Recreates strategy with same trial_count

**Helper function:**
- `restore_optimization_strategy(state)`: Factory to restore any strategy from state

### 3. Worker Integration (`src/web/workers/autotuner_worker.py`)

**On task start (lines ~147-181):**
```python
# Check for existing checkpoint
checkpoint = TaskCheckpoint.load_checkpoint(task.metadata)
if checkpoint:
    # Restore strategy, best_score, iteration from checkpoint
    strategy = restore_optimization_strategy(checkpoint["strategy_state"])
    best_score = checkpoint["best_score"]
    best_experiment_id = checkpoint.get("best_experiment_id")
    iteration = checkpoint["iteration"]
else:
    # Create fresh strategy
    strategy = create_optimization_strategy(optimization_config, task.parameters)
    best_score = float("inf")
    best_experiment_id = None
    iteration = 0
```

**After each experiment (lines ~297-313 and ~331-347):**
```python
await db.commit()

# Save checkpoint after each experiment
try:
    await db.refresh(task)
    updated_metadata = TaskCheckpoint.save_checkpoint(
        task_metadata=task.metadata or {},
        iteration=iteration,
        best_score=best_score,
        best_experiment_id=best_experiment_id,
        strategy_state=strategy.get_state(),
    )
    task.metadata = updated_metadata
    await db.commit()
    logger.info(f"[ARQ Worker] Checkpoint saved at iteration {iteration}")
except Exception as checkpoint_error:
    logger.warning(f"[ARQ Worker] Failed to save checkpoint: {checkpoint_error}")
```

**On task completion (line ~358):**
```python
# Clear checkpoint after successful completion
task.metadata = TaskCheckpoint.clear_checkpoint(task.metadata)
```

## Behavior Changes

### Before Checkpointing:
- Task starts fresh every time
- Timeout = Complete failure, all progress lost
- Manual database cleanup required

### After Checkpointing:
- Task resumes from last completed experiment
- Timeout = Partial progress saved, can resume
- Automatic recovery on next worker invocation

## Usage

No user intervention required - checkpointing is automatic:

1. **Normal execution**: Checkpoints saved but never used (task completes within timeout)
2. **Timeout scenario**:
   - Worker killed by ARQ after 2 hours
   - Task status remains RUNNING
   - Checkpoint exists in task.metadata
   - User restarts task (or worker auto-retries if configured)
   - Worker detects checkpoint and resumes from iteration N
   - Continues until completion or next timeout

## Testing

To test the checkpoint mechanism:

```bash
# 1. Start a long-running task (e.g., 50 iterations)
# 2. Wait for several experiments to complete
# 3. Check database for checkpoint:
sqlite3 ~/.local/share/inference-autotuner/autotuner.db \
  "SELECT id, task_name, json_extract(metadata, '$.checkpoint.iteration') FROM tasks WHERE id=<task_id>;"

# 4. Kill the worker process
pkill -f arq

# 5. Restart worker
./scripts/start_worker.sh

# 6. Trigger task restart (if needed)
# Task should resume from last checkpoint iteration
```

## Limitations

1. **ARQ timeout still applies**: Tasks exceeding 2 hours will be killed
   - Solution: Increase `job_timeout` in `WorkerSettings` (line 336)
   - Or: Split large tasks into smaller chunks

2. **Checkpoint overhead**: Small performance cost to save/load checkpoints
   - Impact: ~100ms per experiment (negligible compared to 10-minute experiments)

3. **Strategy state size**: BayesianStrategy with large history may bloat metadata
   - Mitigation: history is already serialized, JSON compression possible

## Future Enhancements

1. **Automatic retry**: Configure ARQ to auto-retry tasks on timeout
2. **Checkpoint compression**: Compress large strategy states
3. **Checkpoint versioning**: Handle strategy upgrades gracefully
4. **Progress indicators**: Show "Resumed from iteration X" in UI
5. **Checkpoint cleanup**: Prune old checkpoints from failed/cancelled tasks

## Related Files

- `src/web/workers/checkpoint.py` - Checkpoint management
- `src/utils/optimizer.py` - Strategy state serialization
- `src/web/workers/autotuner_worker.py` - Worker integration
- `src/web/db/models.py` - Task metadata field (JSON)

## References

- Issue: Task 7 timeout after 9.6 hours (12/50 experiments completed)
- Solution: Progressive checkpoint save mechanism (Solution 3)
- Implementation date: 2025-11-10
