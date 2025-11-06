# GPU Deployment Strategy for LLM Inference Autotuner

## Overview

This document explains the GPU deployment architecture for the inference autotuner, addressing the Minikube GPU access limitation.

## Problem: Minikube GPU Limitation

### Root Cause

Minikube with Docker driver creates a nested containerization architecture that prevents GPU access:

```
Host (with GPUs) → Docker → Minikube Container → Inner Docker → K8s Pods
                              ↑                      ↑
                              No GPU pass-through    Cannot see GPUs
```

### Evidence

1. **NVIDIA Device Plugin Failure**:
   ```
   E1106 09:17:18.873659 factory.go:112] Incompatible strategy detected auto
   I1106 09:17:18.873686 main.go:381] No devices found. Waiting indefinitely.
   ```

2. **No GPU Resources in Kubernetes**:
   ```bash
   $ kubectl describe node minikube | grep nvidia
   # No nvidia.com/gpu resources found
   ```

3. **Device Plugin Cannot Enumerate GPUs**:
   - Plugin runs successfully in pod
   - Cannot detect any GPU devices
   - Node capacity shows 0 GPUs

### Why This Happens

- **Minikube Docker driver**: Runs Kubernetes in a Docker container
- **No GPU pass-through**: Minikube container doesn't have `--gpus` flag
- **Inner Docker limitation**: Docker-in-Docker cannot inherit GPU access from outer Docker
- **NVIDIA Container Toolkit**: Not available inside Minikube container

### Alternative Solutions Evaluated

#### Option A: Minikube `--driver=none` (NOT RECOMMENDED)

**Pros**:
- Would give Kubernetes direct GPU access
- NVIDIA device plugin would work

**Cons**:
- Requires root privileges
- No isolation (runs K8s directly on host)
- Would disrupt existing Minikube setup
- Risk to other services (ports 3000, 8000 mentioned in requirements)

#### Option B: Full Kubernetes Cluster (FUTURE)

**Pros**:
- Production-ready architecture
- Full GPU support with NVIDIA GPU Operator
- Better isolation and resource management

**Cons**:
- Complex setup (kubeadm, networking, storage)
- Time-consuming for development/testing
- Overkill for current prototype stage

#### Option C: Docker Mode (CURRENT SOLUTION ✅)

**Pros**:
- ✅ Full GPU access with CUDA support
- ✅ Works immediately without infrastructure changes
- ✅ Already implemented in codebase
- ✅ Simpler debugging and faster iteration
- ✅ Same SGLang runtime as Kubernetes deployment

**Cons**:
- ❌ No Kubernetes orchestration (scheduling, autoscaling)
- ❌ No OME CRD benefits (declarative management)

## Solution: Docker Deployment Mode

### Architecture

```
Host Machine
├── Docker Engine (with NVIDIA Container Toolkit)
│   ├── SGLang Container 1 (Experiment 1) → GPU 0
│   ├── SGLang Container 2 (Experiment 2) → GPU 1
│   └── GenAI-Bench (runs locally, connects to containers)
└── Autotuner Orchestrator
    ├── DockerController: Manages containers
    ├── DirectBenchmarkController: Runs benchmarks
    └── Optimizer: Grid search / Bayesian optimization
```

### How It Works

1. **Container Management**:
   - `DockerController` creates and manages SGLang containers
   - Automatic GPU selection based on available memory
   - Port allocation to avoid conflicts
   - Health monitoring via `/health` endpoint

2. **GPU Access**:
   - Containers launched with `--gpus` flag
   - Direct access to host GPUs via NVIDIA Container Toolkit
   - CUDA graphs and FlashAttention fully functional

3. **Benchmarking**:
   - `DirectBenchmarkController` runs genai-bench locally
   - Connects to container via localhost:PORT
   - Measures latency, throughput, TTFT, TPOT
   - Parses results and computes objective scores

4. **Orchestration**:
   - Sequential experiment execution
   - Parameter grid generation
   - Best configuration selection
   - Results persistence

### Verified Capabilities

✅ **Full GPU Utilization**:
- CUDA 12.6.1 detected
- 97GB VRAM per H20 GPU
- KV cache: 64-73GB allocated
- CUDA graphs enabled
- FlashAttention 3 backend

✅ **Performance**:
- Model loading: ~2 seconds
- CUDA graph capture: ~16 seconds
- Latency P50: 176ms, P99: 306ms
- Throughput: 1,730 tokens/s
- 100% success rate on benchmarks

✅ **Autotuning Features**:
- Grid search optimization
- Multiple parameter dimensions
- SLO-aware scoring
- Metrics aggregation
- Result comparison

### Usage

```bash
# Run Docker mode with GPU support
./env/bin/python src/run_autotuner.py examples/docker_task.json --mode docker --verbose

# Task configuration
{
  "task_name": "docker-tune",
  "deployment_mode": "docker",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct"  # Local path or HuggingFace ID
  },
  "base_runtime": "sglang",
  "parameters": {
    "tp-size": [1],
    "mem-fraction-static": [0.7, 0.8, 0.9]
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency"
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "llama-3-2-1b-instruct",
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4, 16]
  }
}
```

## Kubernetes + OME: Non-GPU Capabilities

While Kubernetes cannot access GPUs in the current Minikube setup, the OME environment is still functional for:

### Working Features

✅ **OME Operator**: Fully installed and operational
- Controller manager running (3 replicas)
- All CRDs present and registered
- Webhook validation working

✅ **Resource Management**:
- ClusterBaseModel CRD (model catalog)
- ClusterServingRuntime CRD (runtime definitions)
- InferenceService CRD (service deployments)
- BenchmarkJob CRD (benchmark orchestration)

✅ **CPU-Based Inference** (if needed):
- Can deploy models with CPU backend
- Useful for testing orchestration logic
- Good for development without GPU

### Example: OME InferenceService

```yaml
apiVersion: ome.io/v1beta1
kind: InferenceService
metadata:
  name: llama-cpu-test
  namespace: autotuner
spec:
  model:
    name: llama-3-2-1b-instruct
  engine:
    minReplicas: 1
    maxReplicas: 1
    containers:
      - name: ome-container
        args:
          - --model-path=/mnt/data/models/llama-3-2-1b-instruct
          - --device=cpu
        resources:
          requests:
            cpu: 4
            memory: 8Gi
```

## Recommendations

### For Development and Testing
**Use Docker Mode** (Current)
- Fast iteration
- Full GPU support
- Easier debugging

### For Production Deployment
**Migrate to Real Kubernetes** (Future)
- Setup K8s with kubeadm or managed service (EKS, GKE, AKS)
- Install NVIDIA GPU Operator
- Deploy OME operator
- Use autotuner in OME mode with GPU access

### Migration Path

```
Current:  Docker Mode → SGLang Containers → GPUs
          ↓
          Test features, optimize parameters, validate performance
          ↓
Future:   OME Mode → Kubernetes Pods → GPUs (via GPU Operator)
          ↓
          Production deployment with orchestration, scaling, monitoring
```

## Conclusion

**Current Status**: ✅ **Working GPU-enabled inference autotuning environment**

**Mode**: Docker standalone deployment

**Limitation**: Kubernetes/OME cannot access GPUs in Minikube Docker driver setup

**Recommendation**: Use Docker mode for GPU workloads, maintain OME for orchestration logic and future migration to real Kubernetes

**Next Steps**:
1. Extend Docker mode with more SGLang parameters
2. Add Bayesian optimization support
3. Implement parallel experiment execution
4. Document migration path to GPU-enabled Kubernetes

## References

- [Minikube GPU Support Issue](https://github.com/kubernetes/minikube/issues/8651)
- [NVIDIA Device Plugin Prerequisites](https://github.com/NVIDIA/k8s-device-plugin#prerequisites)
- [Docker Mode Documentation](DOCKER_MODE.md)
- [OME Installation Guide](OME_INSTALLATION.md)
