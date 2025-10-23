# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The LLM Inference Autotuner is a prototype system for automatically tuning LLM inference engine parameters. It supports two deployment modes:

1. **OME Mode**: Kubernetes-based deployment using OME (Open Model Engine)
2. **Docker Mode**: Standalone Docker containers (no Kubernetes required)

The system runs grid search experiments across parameter combinations (e.g., `tp_size`, `mem_frac`), benchmarks each configuration using genai-bench, and identifies the optimal parameters.

## Architecture

### Core Components

**Orchestrator** (`src/run_autotuner.py`)
- Main entry point and task coordinator
- Reads JSON task files from `examples/`
- Manages experiment lifecycle: deploy → benchmark → cleanup
- Supports both OME and Docker modes

**Controllers** (`src/controllers/`)
- All controllers implement `BaseModelController` abstract interface
- **OMEController**: Deploys InferenceServices via Kubernetes CRDs
- **DockerController**: Manages Docker containers directly (GPU-enabled)
- **BenchmarkController**: Kubernetes BenchmarkJob CRD execution
- **DirectBenchmarkController**: Direct genai-bench CLI execution (supports both modes)

**Key Design Pattern**: Strategy pattern for deployment modes
- Docker mode: `DockerController` + `DirectBenchmarkController` (no kubectl)
- OME mode: `OMEController` + `BenchmarkController` OR `DirectBenchmarkController`

### Data Flow

```
Task JSON → Orchestrator → Parameter Grid → For each config:
  1. Controller.deploy_inference_service()
  2. Controller.wait_for_ready()
  3. BenchmarkController.run_benchmark()
  4. Parse metrics & calculate objective score
  5. Controller.delete_inference_service()
→ Find best configuration → Save results/
```

## Running the Autotuner

### Docker Mode (Recommended for Development)
```bash
# Quick test (no Kubernetes needed)
python src/run_autotuner.py examples/docker_task.json --mode docker --direct

# With verbose genai-bench output
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose

# Custom model path
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --model-path /data/models
```

**Note:** Docker containers are automatically removed after they stop for easy cleanup.

### OME Mode (Production)
```bash
# Using Kubernetes BenchmarkJob CRD
python src/run_autotuner.py examples/simple_task.json --mode ome

# Using direct genai-bench CLI (faster, more reliable)
python src/run_autotuner.py examples/simple_task.json --mode ome --direct

# Custom kubeconfig
python src/run_autotuner.py examples/simple_task.json --mode ome --kubeconfig ~/.kube/config
```

## Task Configuration

Task JSON files define experiments. Key fields:

```json
{
  "task_name": "unique-identifier",
  "model": {"name": "model-id", "namespace": "k8s-namespace-or-label"},
  "base_runtime": "sglang" or "llama-3-2-1b-instruct-rt",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2]},
    "mem_frac": {"type": "choice", "values": [0.7, 0.8, 0.9]}
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 10,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "display-name",
    "model_tokenizer": "HuggingFace/model-id",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4],
    "additional_params": {"temperature": 0.0}  // Note: must be numeric, not string
  }
}
```

**Important**: `additional_params` values must be correct types (float 0.0, not string "0.0")

## Installation & Setup

### Dependencies
```bash
pip install -r requirements.txt
# Installs: kubernetes, pyyaml, jinja2, docker, requests
```

### For Docker Mode
```bash
# 1. Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 2. Download model
mkdir -p /mnt/data/models
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /mnt/data/models/llama-3-2-1b-instruct

# 3. Install genai-bench
pip install genai-bench
```

### For OME Mode
```bash
# 1. Install OME (full Kubernetes setup)
./install.sh --install-ome

# 2. Verify OME installation
kubectl get pods -n ome
kubectl get crd | grep ome.io

# 3. Apply model/runtime resources
kubectl apply -f config/examples/
```

## Critical Implementation Details

### Docker Mode Specifics

**GPU Access**: Docker SDK requires:
- Command as list (not string): `command_str.split()`
- DeviceRequest with device_ids: `docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])`
- Do NOT set `CUDA_VISIBLE_DEVICES` env var (conflicts with device_requests)

**Port Management**: Auto-allocates ports 8000-8100 to avoid conflicts between experiments

**Container Lifecycle**: Containers are automatically removed after they stop (equivalent to `docker run --rm`)
- Provides automatic cleanup during experimentation
- Logs not accessible after stop - use `--verbose` flag to capture output during run
- Check running containers: `docker ps`

**Model Path**: Maps host path to `/model` inside container
- Task JSON `model.name` → `/mnt/data/models/{name}` → mounted as `/model`

### OME Mode Specifics

**Templates**: Uses Jinja2 templates in `config/`
- Labels must be strings: `"{{ experiment_id }}"` not `{{ experiment_id }}`
- Environment variables: `$(ENV_VAR)` syntax for K8s expansion

**Benchmark Modes**:
1. **BenchmarkJob CRD**: Uses OME's native job runner (requires working Docker image)
2. **Direct CLI**: Uses local genai-bench with `kubectl port-forward` (recommended)

### Benchmark Execution

**DirectBenchmarkController** has two modes:
- **Docker mode**: Direct URL passed via `endpoint_url` parameter (skips port-forward)
- **OME mode**: Automatic `kubectl port-forward` setup if `endpoint_url=None`

**Verbose Output**: Use `--verbose` flag to stream genai-bench output in real-time
- Useful for debugging connection issues
- Shows progress during long benchmarks (~4 minutes per experiment)

## Common Issues

### Docker Mode
1. **"No accelerator available"**: Command format issue - ensure command is split into list
2. **Model not found**: Check `/mnt/data/models/` path matches task JSON `model.name`
3. **Port conflicts**: Autotuner auto-allocates, but check for existing services on 8000-8100

### OME Mode
1. **InferenceService not ready**: Check `kubectl describe inferenceservice -n autotuner`
2. **GPU resources**: Minikube Docker driver cannot access GPUs (use `--driver=none`)
3. **BenchmarkJob fails**: Use `--direct` mode to bypass genai-bench image issues

### Both Modes
1. **genai-bench parameter errors**: Ensure `additional_params` uses correct types (floats, not strings)
2. **Missing API key**: genai-bench requires `--api-key dummy` even for local servers
3. **Network unreachable**: genai-bench tries to fetch tokenizer from HuggingFace - use offline mode or proxy

## Development Workflow

1. **Test with Docker mode first**: Faster iteration, no Kubernetes overhead
   ```bash
   python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
   ```

2. **Reduce workload for testing**: Edit task JSON to reduce experiments
   ```json
   "max_iterations": 2,
   "max_requests_per_iteration": 10
   ```

3. **Monitor GPU usage**: `nvidia-smi` to check memory allocation
   - Llama-3.2-1B requires ~3GB GPU memory with `mem_frac=0.7`

4. **Check results**: Results saved to `results/{task_name}_results.json`

5. **Inspect benchmark outputs**: `benchmark_results/{task_name}-exp{id}/`

## Project Structure

```
src/
  run_autotuner.py           # Main orchestrator
  controllers/
    base_controller.py       # Abstract interface
    docker_controller.py     # Docker deployment
    ome_controller.py        # Kubernetes/OME deployment
    direct_benchmark_controller.py  # genai-bench CLI execution
    benchmark_controller.py  # K8s BenchmarkJob CRD
  utils/
    optimizer.py             # Parameter grid generation

examples/                    # Task JSON files
  docker_task.json          # Docker mode example
  simple_task.json          # OME mode example

config/                      # K8s resource templates (OME mode)
results/                     # Experiment results (JSON)
benchmark_results/           # genai-bench outputs
docs/                        # Detailed documentation
```

## Documentation

- `README.md`: Installation and usage
- `docs/DOCKER_MODE.md`: Docker deployment guide
- `docs/OME_INSTALLATION.md`: Kubernetes/OME setup
- `docs/GENAI_BENCH_LOGS.md`: Viewing benchmark logs
- `prompts.md`: Development history and troubleshooting notes

## Meta-Instructions

**Important constraints to remember**:
1. Kubernetes Dashboard runs on port 8443 - avoid conflicts
1. Update `prompts.md` when mini-milestones are accomplished
1. Place new `.md` documents in `./docs/`
1. Following further instuctions in `CLAUDE.local.md` if present
