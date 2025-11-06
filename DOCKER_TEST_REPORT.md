# Docker Mode Test Report - GPU Access Verification

## Test Execution Summary

**Date:** November 6, 2025  
**Mode:** Docker (Standalone)  
**Model:** Llama-3.2-1B-Instruct  
**Test Duration:** 128.9 seconds (2 minutes 9 seconds)  
**Status:** ✅ **SUCCESS**

---

## Test Results

### 1. Docker Container Startup: ✅ SUCCESS
- **Container Image:** `lmsysorg/sglang:v0.5.2-cu126`
- **Container 1 (exp1):** Started successfully (ID: 747bde4f3064)
- **Container 2 (exp2):** Started successfully (ID: 7425efd50de7)
- **Ports:** 8002 and 8003 (auto-assigned)
- **Startup Time:** ~30 seconds per container

### 2. GPU Detection and Access: ✅ SUCCESS

**CUDA Version Detected:**
```
CUDA Version 12.6.1
```

**GPU Initialization:**
- GPU 0 successfully passed to containers via `--gpus` flag
- CUDA libraries loaded successfully
- Torch detected CUDA devices properly

**Memory Allocation:**
```
Experiment 1 (mem-fraction-static=0.7):
- Model Load: 2.41 GB
- KV Cache: 63.92 GB (2,094,537 tokens)
- Available after init: 25.62 GB
- Total GPU Memory: ~94.76 GB

Experiment 2 (mem-fraction-static=0.8):
- Model Load: 2.41 GB  
- KV Cache: 73.40 GB (2,405,039 tokens)
- Available after init: 16.12 GB
- Total GPU Memory: ~94.76 GB
```

✅ **Conclusion:** GPU is fully accessible and utilized by Docker containers

### 3. SGLang Server Readiness: ✅ SUCCESS

**Server Initialization Timeline:**
- Model loading: ~2 seconds
- KV cache allocation: ~1 second
- CUDA graph capture: ~16 seconds (35 batch sizes)
- Total warmup: ~30 seconds

**Health Check:**
- Endpoint: `/health`
- Response: 200 OK
- Server ready message: "The server is fired up and ready to roll!"

### 4. GenAI-Bench Execution: ✅ SUCCESS

**Benchmark Configuration:**
- Traffic Pattern: D(100,100) - deterministic 100 input/100 output tokens
- Concurrency Levels: 1 and 4
- Max Requests: 50 per run
- Max Time: 10 seconds per run

**Experiment 1 Results (mem-fraction-static=0.7):**
- Total Requests: 107 (52 @ concurrency=4, 55 @ concurrency=1)
- Success Rate: 100% (0 errors)
- Mean E2E Latency: **0.190 seconds**
- Mean Output Throughput: 1,167.7 tokens/s
- Max Output Throughput: 1,730.6 tokens/s
- P50 Latency: 0.176s
- P99 Latency: 0.359s

**Experiment 2 Results (mem-fraction-static=0.8):**
- Total Requests: 107 (52 @ concurrency=4, 55 @ concurrency=1)
- Success Rate: 100% (0 errors)
- Mean E2E Latency: **0.190 seconds** ⭐ BEST
- Mean Output Throughput: 1,167.4 tokens/s
- Max Output Throughput: 1,730.1 tokens/s
- P50 Latency: 0.176s
- P99 Latency: 0.306s

### 5. Performance Metrics

**GPU Utilization Evidence:**
```
Decode throughput: ~650 tokens/s (single concurrent request)
Decode throughput: ~2,500 tokens/s (batch of 4 requests)
CUDA graphs: Enabled (35 batch sizes captured)
```

**Benchmark Summary:**
| Metric | Exp 1 (0.7) | Exp 2 (0.8) | Winner |
|--------|------------|------------|---------|
| Mean Latency | 0.1902s | 0.1896s | Exp 2 ⭐ |
| Min Latency | 0.1633s | 0.1633s | Tie |
| Max Latency | 0.2172s | 0.2159s | Exp 2 |
| P99 Latency | 0.359s | 0.306s | Exp 2 |
| Success Rate | 100% | 100% | Tie |

**Winner:** Experiment 2 (mem-fraction-static=0.8) with score **0.1896** (lower is better)

---

## Key Observations

### ✅ Successful Aspects

1. **GPU Passthrough Works Perfectly**
   - Docker `--gpus` flag successfully passes GPU to containers
   - No nested containerization issues (unlike Minikube)
   - Full CUDA functionality available

2. **SGLang Performance**
   - CUDA graph optimization working (~650 tokens/s decode)
   - FlashAttention 3 backend selected automatically
   - Efficient memory management (80% vs 70% utilization)

3. **GenAI-Bench Integration**
   - Seamless benchmarking of both configurations
   - Accurate latency and throughput measurements
   - No errors during 214 total requests

4. **Autotuner Orchestration**
   - Automatic port assignment (8002, 8003)
   - Proper container lifecycle management
   - Results aggregation and comparison

### 📊 Performance Insights

1. **Memory Configuration Impact:**
   - 80% memory allocation provides slightly better latency
   - More KV cache tokens: 2.4M vs 2.1M (+14.8%)
   - Trade-off: Less available memory for system operations

2. **Latency Characteristics:**
   - Consistent low latency (~190ms) for 100 token generation
   - Excellent P50 performance (176ms)
   - Good P99 tail latency (306ms for best config)

3. **Throughput Scaling:**
   - Linear scaling from concurrency 1→4
   - Single request: ~650 tokens/s
   - Batched (4): ~2,500 tokens/s (3.8x increase)

---

## Container Logs Analysis

### GPU Detection Logs:
```
CUDA Version 12.6.1
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: 
The pynvml package is deprecated...
```
✅ CUDA properly detected

### Memory Allocation Logs:
```
[2025-11-06 12:10:37] KV Cache is allocated. #tokens: 2405039, 
K size: 36.70 GB, V size: 36.70 GB
[2025-11-06 12:10:37] Memory pool end. avail mem=16.82 GB
```
✅ Large KV cache successfully allocated on GPU

### Performance Logs:
```
[2025-11-06 12:11:03] Decode batch. #running-req: 1, #token: 206, 
token usage: 0.00, cuda graph: True, gen throughput (token/s): 656.23
```
✅ CUDA graphs enabled, high throughput achieved

---

## Comparison: Docker vs Minikube

| Feature | Docker Mode | Minikube Mode |
|---------|-------------|---------------|
| GPU Access | ✅ Direct | ❌ Blocked (nested virt) |
| Setup Complexity | ✅ Simple | ❌ Complex |
| Performance | ✅ Native | N/A (cannot test) |
| Production Ready | ✅ Yes | ⚠️ Dev only |
| Resource Isolation | ✅ Container-level | ✅ Pod-level |

**Recommendation:** Use Docker mode for actual GPU workloads; Minikube is only suitable for CPU testing or development.

---

## Files Generated

1. **Results JSON:** `/root/work/autotuner-ome/results/docker-simple-tune_results.json` (238 KB)
2. **Benchmark Data:** `/root/work/autotuner-ome/benchmark_results/docker-simple-tune-exp{1,2}/`
3. **Container Logs:** Captured in results JSON (65 KB each)
4. **Plots:** Generated by genai-bench in benchmark directories

---

## Conclusion

✅ **Docker mode with GPU support is FULLY FUNCTIONAL**

The test successfully demonstrates that:
- Docker containers can access GPUs via `--gpus` flag
- SGLang server properly utilizes GPU resources
- GenAI-bench accurately measures performance
- The autotuner correctly orchestrates experiments and compares results

**Next Steps:**
1. Test with larger models (7B, 13B parameters)
2. Expand parameter grid (more tp-size, mem-fraction combinations)
3. Test multi-GPU configurations (tp-size > 1)
4. Integrate with production deployment pipelines

---

**Generated:** November 6, 2025 20:11 UTC  
**Test Environment:** Python 3.10, CUDA 12.6.1, SGLang v0.5.2  
**Hardware:** GPU with ~95 GB VRAM (likely A100 80GB or H100)
