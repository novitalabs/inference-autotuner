# Bayesian Optimization

## Overview

Bayesian optimization is an intelligent parameter search strategy that uses probabilistic models to guide the search toward optimal configurations. Unlike grid search which exhaustively evaluates all combinations, Bayesian optimization learns from previous experiments to suggest promising parameter configurations.

## Key Benefits

1. **Sample Efficiency**: Finds good configurations with fewer experiments than grid search
2. **Adaptive Learning**: Uses results from previous experiments to guide future selections
3. **Handles Continuous Parameters**: Naturally supports continuous parameter ranges
4. **Mixed Parameter Types**: Supports categorical, integer, and continuous parameters simultaneously
5. **Exploration-Exploitation Balance**: Balances trying new regions vs exploiting known good areas

## How It Works

### Algorithm

The autotuner uses **Tree-structured Parzen Estimator (TPE)** via the Optuna library:

1. **Initial Random Phase**: First `n_initial_random` trials (default 5) explore randomly
2. **Bayesian Phase**: Subsequent trials use TPE to suggest parameters based on:
   - Performance of previous trials
   - Uncertainty in unexplored regions
   - Balance between exploration and exploitation

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Define Search Space                                      │
│    - Categorical: [val1, val2, val3]                        │
│    - Continuous: {low: 0.5, high: 1.0}                      │
│    - Integer: {low: 1024, high: 16384}                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Random Exploration (n_initial_random trials)             │
│    - Explore search space randomly                          │
│    - Build initial dataset for model                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Bayesian Optimization Loop                               │
│    For each iteration:                                      │
│      a. Fit probabilistic model to past results             │
│      b. Suggest next parameters using acquisition function  │
│      c. Run experiment with suggested parameters            │
│      d. Update model with new result                        │
│      e. Check convergence/stopping criteria                 │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic Bayesian Optimization

```json
{
  "optimization": {
    "strategy": "bayesian",
    "objective": "minimize_latency",
    "max_iterations": 20,
    "n_initial_random": 5
  }
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | string | `"grid_search"` | Set to `"bayesian"` for Bayesian optimization |
| `objective` | string | `"minimize_latency"` | Optimization objective (see below) |
| `max_iterations` | integer | `100` | Maximum number of experiments to run |
| `n_initial_random` | integer | `5` | Number of random trials before Bayesian phase |
| `study_name` | string | auto-generated | Optional Optuna study name |
| `storage` | string | None | Optional Optuna storage URL (e.g., `"sqlite:///optuna.db"`) |

### Supported Objectives

- `minimize_latency`: Minimize end-to-end latency
- `maximize_throughput`: Maximize tokens per second
- `minimize_ttft`: Minimize Time to First Token
- `minimize_tpot`: Minimize Time Per Output Token

## Parameter Specification

### Simple Format (Categorical)

For discrete parameter values, use a simple list:

```json
{
  "parameters": {
    "tensor-parallel-size": [1, 2, 4],
    "schedule-policy": ["lpm", "fcfs"]
  }
}
```

**Note**: These are treated as categorical choices by Bayesian optimization.

### Explicit Format (All Types)

For fine-grained control, use explicit type specification:

#### Categorical Parameters

```json
{
  "param-name": {
    "type": "categorical",
    "values": ["option1", "option2", "option3"]
  }
}
```

#### Continuous Parameters

```json
{
  "mem-fraction-static": {
    "type": "continuous",
    "low": 0.7,
    "high": 0.95
  },
  "learning-rate": {
    "type": "continuous",
    "low": 0.0001,
    "high": 0.1,
    "log": true  // Log scale for parameters like learning rate
  }
}
```

#### Integer Parameters

```json
{
  "max-total-tokens": {
    "type": "integer",
    "low": 4096,
    "high": 16384
  }
}
```

### Mixed Parameter Types Example

```json
{
  "parameters": {
    "tensor-parallel-size": [1, 2, 4],  // Categorical
    "mem-fraction-static": {  // Continuous
      "type": "continuous",
      "low": 0.75,
      "high": 0.95
    },
    "max-total-tokens": {  // Integer
      "type": "integer",
      "low": 4096,
      "high": 16384
    },
    "schedule-policy": ["lpm", "fcfs"],  // Categorical
    "enable-mixed-chunk": [true, false]  // Boolean categorical
  }
}
```

## Complete Example

```json
{
  "task_name": "bayesian-optimization-example",
  "description": "Find optimal parameters using Bayesian optimization",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "runtime_image_tag": "v0.5.2-cu126",
  "parameters": {
    "tensor-parallel-size": [1, 2, 4],
    "mem-fraction-static": {
      "type": "continuous",
      "low": 0.75,
      "high": 0.95
    },
    "max-total-tokens": {
      "type": "integer",
      "low": 4096,
      "high": 16384
    },
    "schedule-policy": ["lpm", "fcfs"],
    "enable-mixed-chunk": [true, false]
  },
  "optimization": {
    "strategy": "bayesian",
    "objective": "minimize_latency",
    "max_iterations": 20,
    "n_initial_random": 5,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "Llama-3.2-1B-Instruct",
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [4],
    "additional_params": {
      "temperature": 0.0,
      "max_tokens": 256
    }
  }
}
```

## Running Bayesian Optimization

### CLI (Direct Mode)

```bash
python src/run_autotuner.py examples/bayesian_task.json --mode docker --direct
```

### Web API

```bash
# Create task
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d @examples/bayesian_task.json

# Start task (returns task_id)
curl -X POST http://localhost:8000/api/tasks/{task_id}/start

# Monitor progress
curl http://localhost:8000/api/tasks/{task_id}
curl http://localhost:8000/api/experiments/task/{task_id}
```

## Comparison: Grid Search vs Bayesian

| Aspect | Grid Search | Bayesian Optimization |
|--------|-------------|----------------------|
| **Strategy** | Exhaustive evaluation | Intelligent sampling |
| **Experiments** | n^p (p=params, n=values) | Configurable (typically 20-50) |
| **Parameter Types** | Discrete only | Categorical, integer, continuous |
| **Adaptability** | Fixed grid | Learns from results |
| **Best For** | Small search spaces, comprehensive testing | Large/continuous spaces, limited budget |
| **Determinism** | Fully deterministic | Stochastic (depends on random seed) |

### Example Comparison

**Search Space**:
- `tensor-parallel-size`: [1, 2, 4] (3 values)
- `mem-fraction-static`: 0.7 to 0.95 (continuous)
- `max-total-tokens`: 4096 to 16384 (continuous)
- `schedule-policy`: ["lpm", "fcfs"] (2 values)

**Grid Search Approach**:
- Would need to discretize continuous parameters
- Example: 3 × 5 × 5 × 2 = 150 experiments
- All experiments run regardless of results

**Bayesian Optimization Approach**:
- Handles continuous parameters naturally
- Typical: 20-30 experiments
- Focuses on promising regions after initial exploration

## Best Practices

### 1. Choose Appropriate Search Space

**Too Narrow**: May miss optimal configuration
```json
{
  "mem-fraction-static": {
    "type": "continuous",
    "low": 0.85,
    "high": 0.90  // Too narrow, only 5% range
  }
}
```

**Better**: Allow wider exploration
```json
{
  "mem-fraction-static": {
    "type": "continuous",
    "low": 0.70,
    "high": 0.95  // 25% range for exploration
  }
}
```

### 2. Set Appropriate Iteration Count

- **Small search space** (< 10 combinations): Use grid search
- **Medium search space** (10-100 combinations): 20-30 Bayesian iterations
- **Large search space** (> 100 combinations): 50-100 Bayesian iterations

### 3. Balance Initial Random Exploration

- **Few parameters** (2-3): `n_initial_random = 3-5`
- **Many parameters** (5+): `n_initial_random = 10-15`
- Rule of thumb: `n_initial_random ≈ 2 × number_of_parameters`

### 4. Use Continuous Parameters When Appropriate

**Discrete Approximation** (grid search style):
```json
{
  "mem-fraction-static": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
}
```

**Continuous** (Bayesian optimization):
```json
{
  "mem-fraction-static": {
    "type": "continuous",
    "low": 0.70,
    "high": 0.95
  }
}
```

The continuous version allows Bayesian optimization to explore values like 0.823 that wouldn't be in a discrete grid.

### 5. Monitor Progress

Check experiment results as they complete to see if optimization is converging:

```bash
# View best score so far
curl http://localhost:8000/api/tasks/{task_id}

# View all experiments sorted by score
curl http://localhost:8000/api/experiments/task/{task_id}
```

## Advanced Features

### Persistent Studies

Store Optuna study to resume interrupted optimization:

```json
{
  "optimization": {
    "strategy": "bayesian",
    "study_name": "llama-3-2-1b-optimization",
    "storage": "sqlite:////root/.local/share/inference-autotuner/optuna.db"
  }
}
```

Benefits:
- Resume optimization if interrupted
- Analyze optimization history with Optuna's visualization tools
- Share study across multiple tasks

### Log-Scale Parameters

For parameters that span multiple orders of magnitude:

```json
{
  "learning-rate": {
    "type": "continuous",
    "low": 0.0001,
    "high": 0.1,
    "log": true  // Sample uniformly in log space
  }
}
```

## Troubleshooting

### Issue: Bayesian optimization performs worse than grid search

**Possible Causes**:
1. Search space too constrained (not enough exploration)
2. Too few iterations (< 20 for typical problems)
3. `n_initial_random` too low (not enough initial data)

**Solutions**:
- Widen parameter ranges
- Increase `max_iterations` to 30-50
- Increase `n_initial_random` to 10-15

### Issue: Optimization not converging

**Symptoms**: Best score not improving after initial trials

**Possible Causes**:
1. Noisy objective function (benchmark variability)
2. Local optima in search space
3. Inappropriate parameter ranges

**Solutions**:
- Run multiple independent optimization runs
- Try different initial seeds
- Review parameter ranges (may be too wide or too narrow)

### Issue: All experiments failing

**Check**:
1. Parameters are valid for the runtime
2. Resource constraints (GPU memory, etc.)
3. Model exists and is accessible
4. Benchmark configuration is correct

## Visualization (Optional)

Optuna provides visualization tools if study is persisted:

```python
import optuna

# Load study
storage = "sqlite:////root/.local/share/inference-autotuner/optuna.db"
study = optuna.load_study(study_name="llama-3-2-1b-optimization", storage=storage)

# Visualize optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Parallel coordinate plot
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Bayesian Optimization Overview](https://arxiv.org/abs/1807.02811)
- [Tree-structured Parzen Estimator](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
