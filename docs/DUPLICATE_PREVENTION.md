# Duplicate Parameter Prevention in Bayesian Optimization

## Overview

The Bayesian optimization strategy now includes automatic duplicate detection and prevention to ensure no two experiments use the exact same parameter combination.

## Problem

In Bayesian optimization, the TPE sampler (Tree-structured Parzen Estimator) can sometimes suggest parameter combinations that have already been tried, especially when:
- The search space is small or discrete
- The optimizer has converged to a local optimum
- Multiple parameters have limited choices

Running duplicate experiments wastes computational resources without providing new information.

## Solution

The `BayesianStrategy` class now:
1. **Tracks all tried parameter combinations** using a set
2. **Detects duplicates** before running an experiment
3. **Applies random perturbation** if a duplicate is detected
4. **Retries** up to 10 times to find a unique combination

## Implementation Details

### Duplicate Detection

```python
# Create hashable representation for duplicate checking
params_tuple = tuple(sorted(params.items()))

# Check if these parameters have been tried before
if params_tuple not in self.tried_params:
    # New parameter combination - use it
    self.tried_params.add(params_tuple)
    return params
```

### Random Perturbation Strategy

When a duplicate is detected, the system applies random perturbation based on parameter type:

**Categorical Parameters:**
- Choose a different value from available choices
- Excludes the current value

**Continuous Parameters:**
- Add random noise: 1-5% of the parameter range
- Direction: randomly positive or negative
- Clamped to valid bounds

**Integer Parameters:**
- Add ±1 or ±2 randomly
- Clamped to valid bounds

### Example Perturbation

Original (duplicate):
```python
{
  "tp-size": 4,
  "mem-fraction-static": 0.85,
  "schedule-policy": "lpm"
}
```

Perturbed:
```python
{
  "tp-size": 4,
  "mem-fraction-static": 0.87,  # Added 2% noise
  "schedule-policy": "lpm"
}
```

## Checkpoint Support

The `tried_params` set is saved in task checkpoints and restored on resume, ensuring duplicate prevention persists across interruptions.

**Checkpoint State:**
```json
{
  "strategy_class": "BayesianStrategy",
  "trial_count": 15,
  "tried_params": [
    [["tp-size", 4], ["mem-fraction-static", 0.85]],
    [["tp-size", 2], ["mem-fraction-static", 0.90]],
    ...
  ]
}
```

## Behavior

### Normal Operation

```
[Bayesian] Trial 5/50: {'tp-size': 4, 'mem-fraction-static': 0.85}
```

### Duplicate Detected

```
[Bayesian] Duplicate detected (attempt 1): {'tp-size': 4, 'mem-fraction-static': 0.85}
[Bayesian] Using perturbed params: {'tp-size': 4, 'mem-fraction-static': 0.87}
```

### Max Attempts Exceeded

If after 10 attempts the system cannot find a unique combination:
```
[Bayesian] Could not find non-duplicate parameters after 10 attempts
```

This triggers early stopping (returns `None`).

## Configuration

No configuration is needed - duplicate prevention is automatic for all Bayesian optimization tasks.

## Benefits

1. **Eliminates wasted computation** - No redundant experiments
2. **Improves search efficiency** - Forces exploration of new regions
3. **Maintains Bayesian integrity** - Perturbations are small enough to preserve optimization quality
4. **Automatic and transparent** - No user intervention required

## Edge Cases

### Small Search Space

For very small discrete search spaces (e.g., 2-3 parameter combinations), the system may exhaust all options before reaching `max_iterations`. This is expected behavior.

### High-Dimensional Spaces

For large continuous or high-dimensional spaces, duplicates are rare and perturbation is rarely needed.

### Checkpoint Resume

When resuming from checkpoint, all previously tried combinations are restored, preventing duplicates across restarts.

## Technical Notes

- Perturbation amount is deliberately small (1-5%) to avoid disrupting Bayesian optimization convergence
- The `tried_params` set uses tuples of sorted key-value pairs for hashability
- Maximum 10 retry attempts balances thoroughness with performance
- Perturbation selects one random parameter to modify per attempt

## Testing

To verify duplicate prevention:

1. Create a task with small discrete search space
2. Set `max_iterations` higher than the number of possible combinations
3. Check logs for "Duplicate detected" messages
4. Verify no two experiments have identical parameters

```bash
# Example: 3 × 3 = 9 possible combinations, but request 20 experiments
{
  "parameters": {
    "tp-size": [1, 2, 4],
    "mem-fraction-static": [0.7, 0.8, 0.9]
  },
  "optimization": {
    "strategy": "bayesian",
    "max_iterations": 20
  }
}
```

Expected: All 9 combinations tried once, then 11 experiments with perturbed parameters.
