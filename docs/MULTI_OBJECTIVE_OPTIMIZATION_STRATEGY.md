# Multi-Objective Optimization Strategy

This document defines the optimization strategies for LLM inference parameter tuning when dealing with multiple conflicting objectives (throughput, latency, TTFT, TPOT).

## Problem Statement

### Challenges

1. **Real experiments are expensive**: Each inference server deployment + benchmark takes 5-10 minutes
2. **Multiple conflicting objectives**: Maximizing throughput vs minimizing latency
3. **Pareto Frontier needed**: Users need to see trade-offs, not just "best" config
4. **Grid search infeasible**: 324 combinations = 27+ hours of testing
5. **Bayesian Optimization limitations**: Designed for single-objective optimization
6. **Aiconfigurator limitations**: Only supports specific models

### Goals

- Reduce experiment count from 300+ to <20 while preserving Pareto Frontier quality
- Support both aiconfigurator-compatible and non-compatible models
- Provide accurate multi-objective trade-off analysis

## Strategy Selection Decision Tree

```
┌─────────────────────────────────────┐
│ Is model supported by aiconfigurator? │
└──────────┬──────────────────────────┘
           │
     ┌─────┴─────┐
     │           │
    YES          NO
     │           │
     ▼           ▼
┌─────────┐  ┌──────────────────┐
│ Two-Stage│  │ Progressive      │
│ Strategy │  │ Pareto Sampling  │
└─────────┘  └──────────────────┘
```

## Strategy 1: Two-Stage Strategy (With Aiconfigurator)

**Use when**: Model is supported by aiconfigurator (LLAMA, QWEN, etc.)

### Overview

Leverage aiconfigurator's static performance model to pre-filter configurations, then run real experiments only on promising candidates.

### Stage 1: Static Prediction (Fast)

```python
# 1. Generate all parameter combinations (e.g., 324 configs)
all_configs = generate_parameter_grid(task.parameters)

# 2. Run aiconfigurator static analysis for all configs (~30 seconds)
predictions = []
for config in all_configs:
    pred = aiconfigurator.predict(
        model=task.model,
        tp=config['tp'],
        pp=config['pp'],
        batch_size=config['batch_size'],
        quantization=config['quantization'],
        # ... other params
    )
    predictions.append({
        'config': config,
        'predicted_throughput': pred.throughput,
        'predicted_ttft': pred.ttft,
        'predicted_tpot': pred.tpot,
        'predicted_latency_p90': pred.latency_p90,
    })

# 3. Compute predicted Pareto Frontier
predicted_frontier = compute_pareto_frontier(
    predictions,
    objectives=['throughput', 'ttft', 'tpot', 'latency_p90'],
    directions=['maximize', 'minimize', 'minimize', 'minimize']
)

# Result: ~10-15 configs on predicted frontier
```

### Stage 2: Real Experiments (Expensive)

```python
# 4. Add extreme points (even if not on predicted frontier)
extreme_configs = [
    min(all_configs, key=lambda c: predictions[c]['ttft']),      # Lowest TTFT
    max(all_configs, key=lambda c: predictions[c]['throughput']), # Highest throughput
    min(all_configs, key=lambda c: predictions[c]['tpot']),      # Lowest TPOT
]

# 5. Combine frontier + extreme points (deduplicate)
configs_to_test = list(set(predicted_frontier + extreme_configs))

# 6. Run real experiments (10-15 configs × 5 min = 50-75 minutes)
real_results = []
for config in configs_to_test:
    result = run_real_experiment(config)
    real_results.append(result)

# 7. Compute actual Pareto Frontier from real results
actual_frontier = compute_pareto_frontier(
    real_results,
    objectives=['throughput', 'ttft', 'tpot', 'latency_p90'],
    directions=['maximize', 'minimize', 'minimize', 'minimize']
)
```

### Expected Savings

- **Without strategy**: 324 experiments × 5 min = 1620 minutes (27 hours)
- **With strategy**: 15 experiments × 5 min = 75 minutes (1.25 hours)
- **Reduction**: 95% fewer experiments

### Implementation Plan

1. Add aiconfigurator prediction step to orchestrator
2. Implement Pareto Frontier computation utility
3. Add "predicted frontier" visualization to frontend
4. Store both predicted and actual metrics in database

## Strategy 2: Progressive Pareto Sampling (Without Aiconfigurator)

**Use when**: Model is NOT supported by aiconfigurator OR user wants pure empirical approach

### Overview

Intelligently sample the parameter space by testing extreme points first, then progressively filling gaps in the frontier.

### Phase 1: Extreme Point Sampling

Test configurations expected to perform well on individual objectives:

```python
# Define extreme configurations based on heuristics
extreme_points = [
    # Lowest latency configuration
    {
        'tp': max(tp_values),           # Higher TP = lower latency
        'batch_size': min(batch_values), # Smaller batch = lower latency
        'quantization': 'fp16',         # No quantization overhead
        'mem_fraction': 0.9,            # More memory = less swapping
    },

    # Highest throughput configuration
    {
        'tp': min(tp_values),           # Lower TP = higher throughput (less comm)
        'batch_size': max(batch_values), # Larger batch = higher throughput
        'quantization': 'fp8',          # Faster computation
        'mem_fraction': 0.9,
    },

    # Lowest TTFT configuration
    {
        'tp': max(tp_values),           # More parallelism
        'prefill_batch': min(prefill_values),
        'enable_chunked_prefill': False,
        'quantization': 'fp16',
    },

    # Lowest TPOT configuration
    {
        'tp': 1,                        # Minimal communication
        'batch_size': 1,
        'quantization': 'fp8',
        'kv_cache_dtype': 'fp8',
    },
]

# Run experiments for extreme points (4 configs)
extreme_results = [run_real_experiment(config) for config in extreme_points]
```

### Phase 2: Gap Filling

Identify gaps in the frontier and test intermediate configurations:

```python
def find_largest_gap(frontier_points, objective1, objective2):
    """Find the largest gap between consecutive points on 2D projection."""
    points = [(p[objective1], p[objective2]) for p in frontier_points]
    points.sort()  # Sort by objective1

    max_gap = 0
    gap_position = None

    for i in range(len(points) - 1):
        gap = euclidean_distance(points[i], points[i+1])
        if gap > max_gap:
            max_gap = gap
            gap_position = i

    return gap_position

# Iteratively fill gaps
current_frontier = extreme_results
max_iterations = 10

for iteration in range(max_iterations):
    # Find largest gap in throughput-latency space
    gap_pos = find_largest_gap(current_frontier, 'throughput', 'latency_p90')

    if gap_pos is None:
        break  # Frontier is dense enough

    # Generate intermediate configuration
    config_a = current_frontier[gap_pos]['config']
    config_b = current_frontier[gap_pos + 1]['config']

    intermediate_config = interpolate_configs(config_a, config_b)

    # Test intermediate configuration
    result = run_real_experiment(intermediate_config)

    # Update frontier if new point is non-dominated
    current_frontier = update_pareto_frontier(current_frontier, result)

    # Stop if frontier hasn't changed in N iterations
    if not_improved_for(3):
        break
```

### Phase 3: Heuristic Refinement

Use performance trends to guide additional sampling:

```python
# Analyze trends from tested configurations
trends = analyze_trends(current_frontier)

# Example trends:
# - "Higher TP always decreases latency"
# - "Batch size > 128 has diminishing returns"
# - "FP8 quantization increases throughput by 30%"

# Generate refined configurations based on trends
refined_configs = []
if trends['tp_improves_latency']:
    refined_configs.append({
        **best_throughput_config,
        'tp': max(tp_values)  # Try higher TP for throughput leader
    })

if trends['fp8_high_benefit']:
    refined_configs.append({
        **best_latency_config,
        'quantization': 'fp8'  # Try FP8 for latency leader
    })

# Test refined configurations
refined_results = [run_real_experiment(config) for config in refined_configs]
final_frontier = update_pareto_frontier(current_frontier, refined_results)
```

### Expected Results

- **Phase 1**: 4 experiments (extreme points)
- **Phase 2**: 5-8 experiments (gap filling)
- **Phase 3**: 3-5 experiments (refinement)
- **Total**: 12-17 experiments (~60-85 minutes)

### Implementation Plan

1. Implement extreme point heuristics for different model types
2. Add Pareto gap detection utility
3. Implement config interpolation logic
4. Add trend analysis module

## Strategy 3: Hybrid Approach (Best of Both Worlds)

Combine aiconfigurator predictions with progressive sampling for maximum efficiency.

### Workflow

```python
# 1. Check if aiconfigurator supports model
if aiconfigurator.supports(task.model):
    # Use two-stage strategy
    predicted_frontier = aiconfigurator.predict_frontier(all_configs)
    configs_to_test = predicted_frontier[:10]  # Top 10
else:
    # Use extreme point sampling
    configs_to_test = generate_extreme_points(task.parameters)

# 2. Run initial experiments
initial_results = [run_real_experiment(c) for c in configs_to_test]

# 3. Compute initial frontier
current_frontier = compute_pareto_frontier(initial_results)

# 4. Progressive refinement (both strategies)
for iteration in range(max_refinement_iterations):
    # Find largest gap
    gap = find_largest_gap(current_frontier)

    # Generate candidate to fill gap
    if aiconfigurator.supports(task.model):
        # Use aiconfigurator to predict best config for gap
        candidate = aiconfigurator.suggest_for_gap(gap, all_configs)
    else:
        # Use interpolation heuristic
        candidate = interpolate_configs(gap.left, gap.right)

    # Test candidate
    result = run_real_experiment(candidate)
    current_frontier = update_pareto_frontier(current_frontier, result)

    if not_improved_for(2):
        break

return current_frontier
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

- [ ] Implement Pareto Frontier computation utility
- [ ] Add multi-objective result comparison
- [ ] Update database schema for predicted vs actual metrics
- [ ] Create visualization for Pareto Frontier

### Phase 2: Two-Stage Strategy (Week 2)

- [ ] Integrate aiconfigurator prediction into orchestrator
- [ ] Add "static analysis" step before experiments
- [ ] Implement frontier filtering logic
- [ ] Add prediction accuracy tracking

### Phase 3: Progressive Sampling (Week 3)

- [ ] Implement extreme point heuristics
- [ ] Add gap detection and filling logic
- [ ] Implement config interpolation
- [ ] Add trend analysis module

### Phase 4: Hybrid Strategy (Week 4)

- [ ] Combine both approaches
- [ ] Add automatic strategy selection
- [ ] Performance tuning and testing
- [ ] Documentation and examples

## Configuration

Add to task JSON:

```json
{
  "optimization": {
    "strategy": "auto",  // "two_stage", "progressive", "hybrid", "auto"
    "objective": "pareto",  // NEW: "pareto" for multi-objective
    "objectives": [
      {
        "name": "throughput",
        "direction": "maximize",
        "weight": 1.0
      },
      {
        "name": "ttft",
        "direction": "minimize",
        "weight": 1.0
      },
      {
        "name": "tpot",
        "direction": "minimize",
        "weight": 1.0
      },
      {
        "name": "latency_p90",
        "direction": "minimize",
        "weight": 1.0
      }
    ],
    "max_experiments": 15,  // Budget constraint
    "frontier_density": 0.1,  // For progressive sampling
    "use_static_predictions": true  // Try aiconfigurator if available
  }
}
```

## Database Schema Updates

```python
class Experiment(Base):
    # ... existing fields ...

    # Predicted metrics (from aiconfigurator)
    predicted_throughput = Column(Float, nullable=True)
    predicted_ttft = Column(Float, nullable=True)
    predicted_tpot = Column(Float, nullable=True)
    predicted_latency_p90 = Column(Float, nullable=True)

    # Actual metrics (from real experiments)
    actual_throughput = Column(Float, nullable=True)
    actual_ttft = Column(Float, nullable=True)
    actual_tpot = Column(Float, nullable=True)
    actual_latency_p90 = Column(Float, nullable=True)

    # Pareto analysis
    is_on_predicted_frontier = Column(Boolean, default=False)
    is_on_actual_frontier = Column(Boolean, default=False)

    # Experiment selection reason
    selection_reason = Column(String, nullable=True)  # "predicted_frontier", "extreme_point", "gap_fill", etc.
```

## Frontend UI Updates

### Pareto Frontier Visualization

```typescript
interface ParetoView {
  // 2D scatter plot
  xAxis: 'throughput' | 'ttft' | 'tpot' | 'latency_p90';
  yAxis: 'throughput' | 'ttft' | 'tpot' | 'latency_p90';

  // Show predicted vs actual
  showPredicted: boolean;
  showActual: boolean;

  // Highlight frontier
  highlightFrontier: boolean;
}
```

### Results Table

| Config | TP | Batch | Quant | Predicted Throughput | Actual Throughput | Predicted TTFT | Actual TTFT | On Frontier | Selection Reason |
|--------|----|----|-------|---------------------|-------------------|----------------|-------------|-------------|-----------------|
| exp-1  | 4  | 128 | fp8   | 1200                | 1150              | 150ms          | 165ms       | ✅ Yes       | Predicted Frontier |
| exp-2  | 1  | 256 | fp8   | 800                 | 820               | 300ms          | 285ms       | ✅ Yes       | Extreme Point |

## Validation Metrics

Track optimization strategy effectiveness:

- **Frontier Coverage**: How many actual frontier points were found?
- **Experiment Efficiency**: Experiments / Actual Frontier Size
- **Prediction Accuracy**: MAE between predicted and actual metrics (for two-stage)
- **Time Savings**: Actual experiments / Total possible experiments

## References

- Aiconfigurator Pareto Analysis: `third_party/aiconfigurator/docs/`
- SLO Scoring: `docs/SLO_SCORING.md`
- Config Schema Alignment: `docs/CONFIG_SCHEMA_ALIGNMENT.md`
