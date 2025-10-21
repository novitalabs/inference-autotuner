# LLM Inference Autotuner - Prototype

Automated parameter tuning for LLM inference engines using OME and genai-bench.

## Project Structure

```
.
├── src/
│   ├── controllers/
│   │   ├── ome_controller.py          # OME InferenceService management
│   │   └── benchmark_controller.py    # genai-bench BenchmarkJob management
│   ├── templates/
│   │   ├── inference_service.yaml.j2  # InferenceService YAML template
│   │   └── benchmark_job.yaml.j2      # BenchmarkJob YAML template
│   ├── utils/
│   │   └── optimizer.py               # Parameter grid generation & scoring
│   └── run_autotuner.py               # Main orchestrator script
├── examples/
│   ├── simple_task.json               # Simple 2x2 parameter grid
│   └── tuning_task.json               # Full parameter grid example
├── third_party/
│   ├── ome/                           # OME submodule
│   └── genai-bench/                   # genai-bench submodule
└── requirements.txt
```

## Prerequisites

1. **Kubernetes cluster** with OME installed
2. **kubectl** configured to access the cluster
3. **Python 3.8+**
4. **Base ServingRuntime** created in the cluster

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize submodules (if not already done)
git submodule update --init --recursive
```

## Usage

### 1. Create a Tuning Task JSON

See `examples/simple_task.json` for the schema:

```json
{
  "task_name": "simple-tune",
  "description": "Description of the tuning task",
  "model": {
    "name": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang-base-runtime",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2]},
    "mem_frac": {"type": "choice", "values": [0.85, 0.9]}
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 4,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4],
    "max_time_per_iteration": 10,
    "max_requests_per_iteration": 50,
    "additional_params": {"temperature": "0.0"}
  }
}
```

### 2. Run the Autotuner

```bash
# Basic usage (uses default kubeconfig)
python src/run_autotuner.py examples/simple_task.json

# Specify kubeconfig path
python src/run_autotuner.py examples/simple_task.json /path/to/kubeconfig
```

### 3. View Results

Results are saved to `results/<task_name>_results.json`

## How It Works

1. **Load Task**: Read JSON configuration file
2. **Generate Parameter Grid**: Create all parameter combinations (grid search)
3. **For Each Configuration**:
   - Deploy InferenceService with parameters
   - Wait for service to be ready
   - Create and run BenchmarkJob
   - Collect metrics
   - Clean up resources
4. **Find Best**: Compare objective scores and report best configuration

## Workflow Example

```
Task: simple-tune (4 combinations: 2 x 2)

Experiment 1: {tp_size: 1, mem_frac: 0.85}
  → Deploy InferenceService
  → Wait for ready
  → Run benchmark
  → Score: 125.3ms

Experiment 2: {tp_size: 1, mem_frac: 0.9}
  → Deploy InferenceService
  → Wait for ready
  → Run benchmark
  → Score: 118.7ms

... (continue for all combinations)

Best: {tp_size: 2, mem_frac: 0.9} → Score: 89.2ms
```

## Configuration Details

### Parameter Types

Currently supported:
- `choice`: List of discrete values

### Optimization Strategies

Currently supported:
- `grid_search`: Exhaustive search over all combinations

### Objectives

Currently supported:
- `minimize_latency`: Minimize average end-to-end latency
- `maximize_throughput`: Maximize tokens/second

## Limitations (Prototype)

- No database persistence (results saved to JSON files)
- No web frontend (uses JSON input files)
- Grid search only (no Bayesian optimization)
- Sequential execution (no parallel experiments)
- Basic error handling
- Simplified metric extraction

## Next Steps

For production implementation:
1. Add database backend (PostgreSQL + InfluxDB)
2. Implement web UI (React + WebSocket)
3. Add Bayesian optimization
4. Enable parallel experiment execution
5. Improve error handling and retry logic
6. Add comprehensive logging
7. Implement metric aggregation and visualization
