# Quick Start Guide

This guide helps you get started with the LLM Inference Autotuner.

## Prerequisites Checklist

Before you begin, ensure you have:

- [ ] **Kubernetes cluster** (v1.28+) with GPU support
- [ ] **OME (Open Model Engine)** installed and running
- [ ] **kubectl** configured to access your cluster
- [ ] **Python 3.8+** with pip
- [ ] At least one **ClusterBaseModel** and **ClusterServingRuntime** configured

## Installation Steps

### Step 1: Install OME (Required)

OME is a **required prerequisite**. If not already installed:

```bash
# See detailed instructions
cat docs/OME_INSTALLATION.md

# Quick verification
kubectl get namespace ome
kubectl get crd | grep ome.io
kubectl get pods -n ome
```

### Step 2: Clone and Setup Autotuner

```bash
# Clone the repository
git clone <repository-url>
cd inference-autotuner

# Run installation script
./install.sh

# The script will:
# - Verify OME is installed (exits if not)
# - Setup Python virtual environment
# - Install dependencies (kubernetes, pyyaml, jinja2)
# - Install genai-bench CLI
# - Create Kubernetes namespace and PVC
# - Verify all installations
```

### Step 3: Configure Your First Tuning Task

```bash
# Edit the example task configuration
vi examples/simple_task.json

# Update:
# - model.name: Must match an existing ClusterBaseModel
# - base_runtime: Must match an existing ClusterServingRuntime
# - parameters: Adjust based on your hardware (e.g., tp_size for GPU count)
```

Example configuration:
```json
{
  "task_name": "my-first-tune",
  "model": {
    "name": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "llama-3-2-1b-instruct-rt",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1]},
    "mem_frac": {"type": "choice", "values": [0.8, 0.85, 0.9]}
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "timeout_per_iteration": 1200
  },
  "benchmark": {
    "task": "text-to-text",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4]
  }
}
```

### Step 4: Run Your First Experiment

```bash
# Activate Python environment
source env/bin/activate

# Run autotuner in direct CLI mode (recommended)
python src/run_autotuner.py examples/simple_task.json --direct

# Or use Kubernetes BenchmarkJob mode
python src/run_autotuner.py examples/simple_task.json
```

### Step 5: View Results

```bash
# Results are saved to results/<task_name>_results.json
cat results/my-first-tune_results.json

# Example output:
# {
#   "task_name": "my-first-tune",
#   "total_experiments": 3,
#   "successful_experiments": 3,
#   "best_result": {
#     "experiment_id": 2,
#     "parameters": {"tp_size": 1, "mem_frac": 0.85},
#     "metrics": {
#       "latency_ms": 125.3,
#       "throughput_tps": 89.2
#     }
#   }
# }
```

## Common Issues

### Issue: "OME namespace not found"

**Solution:** Install OME first. See `docs/OME_INSTALLATION.md`

### Issue: "No ClusterBaseModels found"

**Solution:** Create a model resource:
```bash
kubectl apply -f docs/examples/model-llama-3.2-1b.yaml
```

See OME installation guide for examples.

### Issue: "InferenceService not becoming Ready"

**Debug:**
```bash
# Check InferenceService status
kubectl describe inferenceservice <name> -n autotuner

# Check pods
kubectl get pods -n autotuner

# Check logs
kubectl logs <pod-name> -n autotuner
```

## Benchmark Execution Modes

The autotuner supports two modes:

### 1. Direct CLI Mode (Recommended)
- Uses local genai-bench installation
- Automatic port-forwarding
- Faster and more reliable

```bash
python src/run_autotuner.py examples/simple_task.json --direct
```

### 2. Kubernetes BenchmarkJob Mode
- Uses OME BenchmarkJob CRD
- Runs benchmarks in Kubernetes pods
- Requires working genai-bench Docker image

```bash
python src/run_autotuner.py examples/simple_task.json
```

## Next Steps

- **Explore Parameters**: Add more parameters to tune (batch_size, schedule_policy, etc.)
- **Try Different Models**: Test with different ClusterBaseModels
- **Optimize Objectives**: Switch between minimize_latency and maximize_throughput
- **Scale Up**: Run larger experiments with more parameter combinations

## Documentation

- **Main README**: [README.md](README.md)
- **OME Installation**: [docs/OME_INSTALLATION.md](docs/OME_INSTALLATION.md)
- **Installation Summary**: [INSTALL_SUMMARY.md](INSTALL_SUMMARY.md)
- **Troubleshooting**: See README.md Troubleshooting section

## Getting Help

If you encounter issues:

1. Check logs: `kubectl logs -n ome deployment/ome-controller-manager`
2. Verify environment: See README.md "Environment Verification" section
3. Review troubleshooting guides in documentation
4. Check OME GitHub issues: https://github.com/sgl-project/ome/issues

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Verify OME is installed
kubectl get namespace ome
kubectl get clusterbasemodels
kubectl get clusterservingruntimes

# 2. Install autotuner
./install.sh

# 3. Activate environment
source env/bin/activate

# 4. List available resources
kubectl get clusterbasemodels
kubectl get clusterservingruntimes

# 5. Edit task configuration
vi examples/simple_task.json
# Update model and runtime names to match available resources

# 6. Run experiment
python src/run_autotuner.py examples/simple_task.json --direct

# 7. View results
cat results/simple-tune_results.json

# 8. Monitor running experiments
kubectl get inferenceservices -n autotuner -w
```

That's it! You're now ready to start tuning LLM inference parameters automatically.
