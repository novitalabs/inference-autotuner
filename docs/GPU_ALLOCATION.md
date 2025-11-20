# Intelligent GPU Allocation for OME Deployments

## Overview

The LLM Inference Autotuner now includes intelligent GPU allocation for OME (Kubernetes) deployments. When deploying inference services, the system automatically discovers cluster-wide GPU resources and selects the most suitable node based on real-time utilization metrics.

## Features

### 1. Cluster-wide GPU Discovery

The system queries all nodes in the Kubernetes cluster to collect:
- GPU capacity and allocatable resources
- Real-time utilization metrics (when available)
- Memory usage per GPU
- Temperature readings
- Node-level GPU availability

### 2. Intelligent Node Selection

When deploying a new InferenceService, the controller:

1. **Determines GPU requirements** from task parameters (`tp-size`, `tp_size`, or `tpsize`)
2. **Queries cluster GPU status** across all nodes
3. **Ranks nodes** based on:
   - Availability of idle GPUs (< 30% utilization AND < 50% memory)
   - Number of idle GPUs available
   - Average utilization across the node
4. **Selects the best node** with the most idle resources
5. **Applies node affinity** to the InferenceService YAML

### 3. Automatic Fallback

- If no GPU metrics are available (no running pods on a node), all allocatable GPUs are considered idle
- If no idle GPUs are found, the system falls back to Kubernetes scheduler's default behavior
- Node selection can be disabled by setting `enable_gpu_selection=False`

## Architecture

### GPU Discovery Module

**File**: `src/utils/gpu_discovery.py`

Key components:
- `get_cluster_gpu_status()`: Queries all GPU nodes and collects metrics
- `get_node_gpu_summaries()`: Aggregates GPUs by node with statistics
- `find_best_node_for_deployment()`: Implements selection algorithm

### OME Controller Integration

**File**: `src/controllers/ome_controller.py`

Changes:
- Added `enable_gpu_selection` parameter to `deploy_inference_service()`
- Integrated GPU discovery before deployment
- Passes selected node to Jinja2 template

### InferenceService Template

**File**: `src/templates/inference_service.yaml.j2`

Added conditional node affinity:
```yaml
{% if selected_node %}
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - {{ selected_node }}
{% endif %}
```

## Usage

### Automatic (Default Behavior)

GPU allocation is enabled by default for all OME deployments:

```python
from src.controllers.ome_controller import OMEController

controller = OMEController()

# GPU selection happens automatically
controller.deploy_inference_service(
    task_name="my-task",
    experiment_id="1",
    namespace="autotuner",
    model_name="llama-3-2-1b-instruct",
    runtime_name="sglang-llama-small",
    parameters={"tp-size": 1}
)
```

Example output:
```
=== GPU Node Selection ===
Looking for node with 1 idle GPU(s)...
Selected node 'host-10-97-65-155' for deployment:
  - Allocatable GPUs: 8/8
  - Idle GPUs: 8
  - Avg Utilization: 0.0%
  - Avg Memory Usage: 0.0%
✓ Selected node: host-10-97-65-155
==========================

Created InferenceService 'my-task-exp1' in namespace 'autotuner' on node 'host-10-97-65-155'
```

### Manual Node Selection (Disabled)

To disable automatic GPU selection:

```python
controller.deploy_inference_service(
    task_name="my-task",
    experiment_id="1",
    namespace="autotuner",
    model_name="llama-3-2-1b-instruct",
    runtime_name="sglang-llama-small",
    parameters={"tp-size": 1},
    enable_gpu_selection=False  # Use Kubernetes scheduler
)
```

### Querying GPU Status Directly

```python
from src.utils.gpu_discovery import (
    get_cluster_gpu_status,
    get_node_gpu_summaries,
    find_best_node_for_deployment
)

# Get all GPU information
gpus = get_cluster_gpu_status()
for gpu in gpus:
    print(f"{gpu.node_name} GPU {gpu.gpu_index}: {gpu.utilization_percent}%")

# Get node summaries
summaries = get_node_gpu_summaries()
for node_name, summary in summaries.items():
    print(f"{node_name}: {summary.idle_gpu_count} idle GPUs")

# Find best node for deployment
best_node = find_best_node_for_deployment(required_gpus=2)
print(f"Best node: {best_node}")
```

## Selection Criteria

### GPU "Idle" Definition

A GPU is considered **idle** if:
- Utilization < 30% AND
- Memory usage < 50%

OR

- No metrics available AND allocatable (no running pods on node)

### Node Ranking Priority

Nodes are ranked by the following criteria (in order):

1. **Has idle GPUs**: Prefer nodes with at least `required_gpus` idle GPUs
2. **Idle GPU count**: Among nodes with idle GPUs, prefer those with more idle resources
3. **Average utilization**: If no idle GPUs, prefer node with lowest utilization

### Configurable Thresholds

You can customize the idle thresholds when calling `find_best_node_for_deployment()`:

```python
best_node = find_best_node_for_deployment(
    required_gpus=2,
    utilization_threshold=20.0,  # Default: 30.0
    memory_threshold=40.0         # Default: 50.0
)
```

## Metrics Collection

### Local Node (API Server Host)

- Uses direct `nvidia-smi` execution
- Fast and always available
- Full metrics: memory, utilization, temperature

### Remote Nodes

- Finds pods with GPU resource requests (`nvidia.com/gpu`)
- Executes `kubectl exec` into GPU-enabled pods
- Runs `nvidia-smi` remotely
- Falls back to "no metrics" if no suitable pods found

### No Metrics Available

When no GPU metrics can be collected for a node:
- All allocatable GPUs are considered idle
- Suitable for nodes with no running workloads
- Ensures new deployments can still be scheduled

## Benefits

1. **Load Balancing**: Distributes workloads across nodes with idle resources
2. **Reduced Contention**: Avoids scheduling on nodes with high GPU usage
3. **Improved Performance**: Experiments run on less-loaded GPUs perform better
4. **Automatic Failover**: Falls back to Kubernetes scheduler if selection fails
5. **Transparent**: Clear logging shows node selection reasoning

## Troubleshooting

### No Suitable Node Found

If the controller can't find a suitable node:

1. Check that GPU nodes exist:
   ```bash
   kubectl get nodes -o json | jq '.items[] | select(.status.capacity["nvidia.com/gpu"] != null) | {name: .metadata.name, gpus: .status.capacity["nvidia.com/gpu"]}'
   ```

2. Verify GPU metrics collection:
   ```python
   python3 -c "from src.utils.gpu_discovery import get_node_gpu_summaries; print(get_node_gpu_summaries())"
   ```

3. Check for pods with GPU access on remote nodes:
   ```bash
   kubectl get pods --all-namespaces -o json | jq '.items[] | select(.spec.containers[].resources.limits["nvidia.com/gpu"] != null) | {name: .metadata.name, node: .spec.nodeName}'
   ```

### InferenceService Not Scheduled

If an InferenceService is created but pod remains pending:

1. Check node affinity was applied:
   ```bash
   kubectl get inferenceservice -n autotuner <name> -o yaml | grep -A 10 affinity
   ```

2. Verify the selected node has allocatable GPUs:
   ```bash
   kubectl describe node <selected-node> | grep nvidia.com/gpu
   ```

3. Check for resource conflicts or taints:
   ```bash
   kubectl describe node <selected-node> | grep -A 5 Taints
   ```

### Metrics Not Available

If GPU metrics are consistently unavailable:

1. Deploy a test pod with GPU access:
   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: gpu-test
   spec:
     containers:
     - name: cuda
       image: nvidia/cuda:11.8.0-base-ubuntu22.04
       command: ["sleep", "infinity"]
       resources:
         limits:
           nvidia.com/gpu: 1
   ```

2. Verify nvidia-smi works in the pod:
   ```bash
   kubectl exec gpu-test -- nvidia-smi
   ```

## Future Enhancements

Potential improvements for future versions:

- **Historical metrics**: Track GPU usage over time for better predictions
- **Multi-constraint scheduling**: Consider CPU, memory, and storage alongside GPU
- **Gang scheduling**: Reserve multiple GPUs on the same node for multi-GPU tasks
- **Preference-based selection**: Allow users to specify node preferences
- **GPU fragmentation awareness**: Prefer nodes with contiguous free GPUs
- **Integration with Prometheus**: Use cluster-wide GPU metrics exporters

## References

- Kubernetes Node Affinity: https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity
- NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
- OME InferenceService CRD: https://github.com/sgl-project/ome
