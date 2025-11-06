# Quick Start Guide - Verified Commands

## Test the GPU-Enabled Autotuner (Docker Mode)

### Quick Test (2 experiments, ~5-10 minutes)
```bash
cd /root/work/autotuner-ome
./env/bin/python src/run_autotuner.py examples/docker_task.json --mode docker --verbose
```

**What this tests:**
- ✅ Docker container deployment with GPU
- ✅ SGLang server with CUDA support
- ✅ Model loading and inference
- ✅ GenAI-bench benchmarking
- ✅ Parameter optimization (2 configs)
- ✅ Results aggregation

**Expected output:**
```
[Docker] Using local model at /mnt/data/models/llama-3-2-1b-instruct
[Docker] Container started (ID: abc123...)
[Docker] Service is ready! URL: http://localhost:8002
[Benchmark] Running genai-bench for experiment 1
...
Best experiment: #2 with score 0.1896
Results saved to: results/docker-simple-tune_results.json
```

---

## Test OME Orchestration Layer

### Verify OME Integration (creates/deletes InferenceService)
```bash
cd /root/work/autotuner-ome
./env/bin/python scripts/test_ome_basic.py
```

**What this tests:**
- ✅ OME API server connectivity
- ✅ InferenceService CRD functionality
- ✅ Resource creation via Kubernetes API
- ✅ Status tracking and retrieval
- ✅ Resource deletion and cleanup

**Expected output:**
```
Testing OME Orchestration...
✅ InferenceService created successfully
   Name: test-ome-orchestration
   Namespace: autotuner
✅ InferenceService retrieved
✅ InferenceService deleted
OME Orchestration Test Complete!
```

---

## Monitor GPU Usage

### During Docker Mode Tests
```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Check specific container logs
docker logs autotuner-docker-simple-tune-exp1

# View benchmark results
cat results/docker-simple-tune_results.json | jq '.best_experiment'
```

---

## Verify Environment

### Check All Components
```bash
# GPUs available
nvidia-smi --query-gpu=name,memory.free --format=csv

# Python environment
./env/bin/python --version  # Should be 3.10.12
./env/bin/genai-bench --version  # Should be 0.0.2

# Kubernetes cluster
kubectl cluster-info
kubectl get pods -n ome  # OME controllers should be Running

# OME resources
kubectl get clusterbasemodels
kubectl get clusterservingruntimes
```

---

## Results Location

```bash
# Autotuner results
ls -lh results/
cat results/docker-simple-tune_results.json

# Benchmark raw data
ls -lh benchmark_results/

# Container logs (after test)
docker ps -a | grep autotuner
```

---

## Common Issues

### GPU Out of Memory
```bash
# Check GPU usage first
nvidia-smi

# Use different GPU or lower mem-fraction
# Edit examples/docker_task.json:
"mem-fraction-static": [0.6, 0.7]  # Lower values
```

### Port Already in Use
```bash
# Check what's using the port
netstat -tlnp | grep 8000

# The autotuner auto-selects ports 8000-8100
# It will automatically find available port
```

### GenAI-bench Not Found
```bash
# Verify installation
./env/bin/pip list | grep genai-bench

# Reinstall if needed
./env/bin/pip install genai-bench
```

---

## Full Test Suite

### Run All Tests
```bash
cd /root/work/autotuner-ome

# 1. OME orchestration test
echo "=== Testing OME Orchestration ==="
./env/bin/python scripts/test_ome_basic.py

# 2. Docker mode with GPU
echo "=== Testing Docker Mode with GPU ==="
./env/bin/python src/run_autotuner.py examples/docker_task.json --mode docker --verbose

# 3. Check results
echo "=== Results ==="
ls -lh results/
cat results/docker-simple-tune_results.json | jq '.task_summary'
```

---

## Quick Reference

| Command | Purpose | Duration |
|---------|---------|----------|
| `test_ome_basic.py` | Verify OME works | <5 sec |
| `docker_task.json --mode docker` | GPU inference test | 5-10 min |
| `nvidia-smi` | Check GPU status | Instant |
| `kubectl get pods -n ome` | Check OME status | Instant |

---

## Documentation Reference

- **Task Completion**: `docs/TASK_COMPLETION_SUMMARY.md`
- **Agent Log**: `docs/agentlog-ome.md`
- **GPU Strategy**: `docs/GPU_DEPLOYMENT_STRATEGY.md`
- **OME Test Report**: `docs/OME_ORCHESTRATION_TEST_REPORT.md`
- **Testing Summary**: `TESTING_SUMMARY.md`
- **Docker Test**: `DOCKER_TEST_REPORT.md`

---

## Success Criteria

Your test is successful if you see:

1. ✅ Docker containers start with GPU access
2. ✅ SGLang loads model and serves requests
3. ✅ GenAI-bench completes benchmarks
4. ✅ Results JSON file created with metrics
5. ✅ No errors in execution
6. ✅ Success rate = 100%

**Example Best Result:**
```json
{
  "experiment_id": 2,
  "parameters": {"mem-fraction-static": 0.8},
  "score": 0.1896,
  "metrics": {
    "p50_e2e_latency": 0.176,
    "max_output_throughput": 1730.35,
    "success_rate": 1.0
  }
}
```

This confirms the entire GPU-enabled autotuning pipeline is working!
