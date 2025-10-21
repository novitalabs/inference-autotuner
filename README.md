# LLM Inference Autotuner - Prototype

Automated parameter tuning for LLM inference engines using OME and genai-bench.

## Project Structure

```
.
├── src/
│   ├── controllers/
│   │   ├── ome_controller.py                  # OME InferenceService management
│   │   ├── benchmark_controller.py            # genai-bench BenchmarkJob management (K8s CRD)
│   │   └── direct_benchmark_controller.py     # Direct genai-bench CLI execution
│   ├── templates/
│   │   ├── inference_service.yaml.j2          # InferenceService YAML template
│   │   └── benchmark_job.yaml.j2              # BenchmarkJob YAML template
│   ├── utils/
│   │   └── optimizer.py                       # Parameter grid generation & scoring
│   └── run_autotuner.py                       # Main orchestrator script
├── examples/
│   ├── simple_task.json                       # Simple 2x2 parameter grid
│   └── tuning_task.json                       # Full parameter grid example
├── third_party/
│   ├── ome/                                   # OME submodule
│   └── genai-bench/                           # genai-bench submodule
├── env/                                       # Python virtual environment
│   └── bin/genai-bench                        # genai-bench CLI executable
└── requirements.txt
```

## Prerequisites

### Environment Requirements

1. **Kubernetes cluster** (v1.28+) with OME installed
   - Tested on Minikube v1.34.0
   - Single-node or multi-node cluster

2. **OME Operator** (Open Model Engine)
   - Version: v0.1.3 or later
   - Installed in `ome` namespace
   - All CRDs must be present: `inferenceservices`, `benchmarkjobs`, `clusterbasemodels`, `clusterservingruntimes`

3. **kubectl** configured to access the cluster

4. **Python 3.8+** with pip

5. **Model and Runtime Resources**
   - At least one `ClusterBaseModel` available
   - At least one `ClusterServingRuntime` configured
   - Example: `llama-3-2-1b-instruct` model with `llama-3-2-1b-instruct-rt` runtime

### Environment Verification

Run these commands to verify your environment:

```bash
# Check Kubernetes connection
kubectl cluster-info

# Check OME installation
kubectl get pods -n ome
kubectl get crd | grep ome.io

# Check available models and runtimes
kubectl get clusterbasemodels
kubectl get clusterservingruntimes

# Verify resources
kubectl describe node | grep -A 5 "Allocated resources"
```

Expected output:
- OME controller pods running
- CRDs: `inferenceservices.ome.io`, `benchmarkjobs.ome.io`, etc.
- At least one model in Ready state
- At least one runtime available

## Installation

### 1. Clone Repository and Submodules

```bash
git clone <repository-url>
cd inference-autotuner

# Initialize submodules
git submodule update --init --recursive
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `kubernetes>=28.1.0` - K8s Python client
- `pyyaml>=6.0` - YAML parsing
- `jinja2>=3.1.0` - Template rendering

### 3. Create Required Resources

#### a) Create Namespace

```bash
kubectl create namespace autotuner
```

#### b) Create PersistentVolumeClaim for Benchmark Results

```bash
kubectl apply -f config/benchmark-pvc.yaml
```

This creates a 1Gi PVC named `benchmark-results-pvc` where benchmark results will be stored.

### 4. Configure Task JSON

Update `examples/simple_task.json` or `examples/tuning_task.json` with:
- Correct model name (must match a `ClusterBaseModel`)
- Correct runtime name (must match a `ClusterServingRuntime`)
- Parameters appropriate for your hardware (e.g., `tp_size` limited by GPU count)

Example:
```json
{
  "model": {
    "name": "llama-3-2-1b-instruct",  // Must exist as ClusterBaseModel
    "namespace": "autotuner"
  },
  "base_runtime": "llama-3-2-1b-instruct-rt",  // Must exist as ClusterServingRuntime
  "parameters": {
    "tp_size": {"type": "choice", "values": [1]},  // Adjust based on GPU count
    "mem_frac": {"type": "choice", "values": [0.8, 0.85, 0.9]}
  }
}
```

## Usage

### Benchmark Execution Modes

The autotuner supports two benchmark execution modes:

1. **Kubernetes BenchmarkJob Mode** (Default):
   - Uses OME's BenchmarkJob CRD
   - Runs genai-bench in Kubernetes pods
   - Requires working genai-bench Docker image
   - More complex but native to OME

2. **Direct CLI Mode** (Recommended):
   - Runs genai-bench directly using local installation
   - Automatic port forwarding to InferenceService
   - Bypasses Docker image issues
   - Faster and more reliable for prototyping

### 1. Direct CLI Mode (Recommended)

Run benchmarks using the local genai-bench installation:

```bash
python src/run_autotuner.py examples/simple_task.json --direct
```

**How it works:**
- Deploys InferenceService via OME
- Automatically sets up `kubectl port-forward` to access the service
- Runs genai-bench CLI directly from `env/bin/genai-bench`
- Cleans up port forward after completion
- No Docker image dependencies

**Requirements:**
- genai-bench installed in Python environment (`pip install genai-bench`)
- `kubectl` configured and accessible
- No additional configuration needed

### 2. Kubernetes BenchmarkJob Mode

Run benchmarks using OME's BenchmarkJob CRD:

```bash
python src/run_autotuner.py examples/simple_task.json
```

**How it works:**
- Creates Kubernetes BenchmarkJob resources
- Uses genai-bench Docker image
- Results stored in PersistentVolumeClaim

**Requirements:**
- PVC created (see installation step 3b)
- Working genai-bench Docker image accessible to cluster

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

## Troubleshooting

### Common Issues and Solutions

#### 1. InferenceService Creation Fails: "cannot unmarshal number into Go struct field"

**Error:**
```
cannot unmarshal number into Go struct field ObjectMeta.labels of type string
```

**Cause:** Label values in Kubernetes must be strings, but numeric values were provided.

**Solution:** Already fixed in templates. Labels are now quoted:
```yaml
labels:
  autotuner.io/experiment-id: "{{ experiment_id }}"  # Quoted
```

#### 2. Deployment Fails: "spec.template.spec.containers[0].name: Required value"

**Error:**
```
spec.template.spec.containers[0].name: Required value
```

**Cause:** Container name was missing in the runner specification.

**Solution:** Already fixed. Template now includes:
```yaml
runner:
  name: ome-container  # Required field
```

#### 3. SGLang Fails: "Can't load the configuration of '$MODEL_PATH'"

**Error:**
```
OSError: Can't load the configuration of '$MODEL_PATH'
```

**Cause:** Environment variable not being expanded in the args list.

**Solution:** Already fixed. Template uses K8s env var syntax:
```yaml
args:
  - --model-path
  - $(MODEL_PATH)  # Proper K8s env var expansion
```

#### 4. BenchmarkJob Creation Fails: "spec.outputLocation: Required value"

**Error:**
```
spec.outputLocation: Required value
```

**Cause:** OME BenchmarkJob CRD requires an output storage location.

**Solution:** Already fixed. Template includes:
```yaml
outputLocation:
  storageUri: "pvc://benchmark-results-pvc/{{ benchmark_name }}"
```

Make sure the PVC exists:
```bash
kubectl apply -f config/benchmark-pvc.yaml
```

#### 5. BenchmarkJob Fails: "unknown storage type for URI: local:///"

**Error:**
```
unknown storage type for URI: local:///tmp/...
```

**Cause:** OME only supports `pvc://` (Persistent Volume Claims) and `oci://` (Object Storage).

**Solution:** Use PVC storage (already configured):
```bash
# Create PVC first
kubectl apply -f config/benchmark-pvc.yaml

# Template automatically uses pvc://benchmark-results-pvc/
```

#### 6. InferenceService Not Becoming Ready

**Symptoms:**
- InferenceService shows `Ready=False`
- Status: "ComponentNotReady: Target service not ready for ingress creation"

**Debugging Steps:**
```bash
# Check pod status
kubectl get pods -n autotuner

# Check pod logs
kubectl logs <pod-name> -n autotuner --tail=50

# Check InferenceService events
kubectl describe inferenceservice <name> -n autotuner
```

**Common Causes:**
- Model not found or not ready
- Runtime mismatch with model
- Insufficient GPU resources
- Container image pull errors

**Typical Wait Time:** 60-90 seconds for model loading and CUDA graph capture

#### 7. Wrong Model or Runtime Name

**Symptoms:**
- InferenceService fails to create
- Error about model or runtime not found

**Solution:**
```bash
# List available models
kubectl get clusterbasemodels

# List available runtimes
kubectl get clusterservingruntimes

# Update examples/simple_task.json with correct names
```

#### 8. BenchmarkJob Stays in "Running" Status

**Symptoms:**
- BenchmarkJob doesn't complete
- No error messages

**Debugging:**
```bash
# Check benchmark pod logs
kubectl get pods -n autotuner | grep bench
kubectl logs <benchmark-pod> -n autotuner

# Check BenchmarkJob status
kubectl describe benchmarkjob <name> -n autotuner
```

**Common Causes:**
- InferenceService endpoint not reachable
- Traffic scenarios too demanding
- Timeout settings too low

### Monitoring Tips

**Watch resources in real-time:**
```bash
# All resources in autotuner namespace
watch kubectl get inferenceservices,benchmarkjobs,pods -n autotuner

# Just InferenceServices
kubectl get inferenceservices -n autotuner -w

# Pod logs
kubectl logs -f <pod-name> -n autotuner
```

**Check OME controller logs:**
```bash
kubectl logs -n ome deployment/ome-controller-manager --tail=100
```

### Performance Tips

1. **Reduce timeout values** for faster iteration during development:
   ```json
   "optimization": {
     "timeout_per_iteration": 300  // 5 minutes instead of 10
   }
   ```

2. **Use smaller benchmark workloads** for testing:
   ```json
   "benchmark": {
     "traffic_scenarios": ["D(100,100)"],  // Lighter load
     "max_requests_per_iteration": 50      // Fewer requests
   }
   ```

3. **Limit parameter grid** for initial testing:
   ```json
   "parameters": {
     "mem_frac": {"type": "choice", "values": [0.85, 0.9]}  // Just 2 values
   }
   ```

## Next Steps

For production implementation:
1. Add database backend (PostgreSQL + InfluxDB)
2. Implement web UI (React + WebSocket)
3. Add Bayesian optimization
4. Enable parallel experiment execution
5. Improve error handling and retry logic
6. Add comprehensive logging
7. Implement metric aggregation and visualization
