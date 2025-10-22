# SGLang Runtime Metrics and Parameters

This document provides detailed information about SGLang runtime configuration, tunable parameters, and available metrics for the inference autotuner.

## Table of Contents

1. [Runtime Overview](#runtime-overview)
2. [Tunable Parameters](#tunable-parameters)
3. [Performance Metrics](#performance-metrics)
4. [Parameter Tuning Guide](#parameter-tuning-guide)
5. [Metrics Collection](#metrics-collection)

---

## Runtime Overview

**SGLang** (SGLang: Efficient Execution of Structured Language Programs) is a high-performance inference engine for large language models developed by the LMSYS team.

### Key Features
- **Optimized Performance**: Faster than vLLM in many scenarios
- **Structured Generation**: Native support for constrained decoding
- **Multi-GPU Support**: Tensor parallelism and pipeline parallelism
- **Memory Efficiency**: Advanced KV cache management
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API

### Version in Use
- **Image**: `docker.io/lmsysorg/sglang:v0.5.2-cu126`
- **CUDA**: 12.6
- **Protocol**: OpenAI-compatible

---

## Tunable Parameters

These parameters can be varied by the autotuner to optimize inference performance:

### 1. Tensor Parallelism Size (`--tp-size`)

**Description**: Number of GPUs to use for tensor parallelism

**Parameter**: `tp_size`

**Values**:
- Type: Integer
- Range: 1 to number of available GPUs
- Default: 1

**Impact**:
- **Higher values**: 
  - ✅ Can handle larger models
  - ✅ May improve throughput for large batch sizes
  - ❌ Requires more GPUs
  - ❌ Increased communication overhead
- **Lower values**:
  - ✅ More efficient for smaller models
  - ✅ Less resource usage
  - ❌ Model may not fit in memory

**Use Cases**:
- `tp_size=1`: Models ≤ 13B parameters (single GPU)
- `tp_size=2`: Models 13B-70B parameters
- `tp_size=4`: Models 70B-180B parameters
- `tp_size=8`: Models ≥ 180B parameters

### 2. GPU Memory Fraction (`--mem-frac`)

**Description**: Fraction of GPU memory to use for KV cache

**Parameter**: `mem_frac`

**Values**:
- Type: Float
- Range: 0.0 to 1.0
- Default: 0.9
- Recommended: 0.80 - 0.95

**Impact**:
- **Higher values (0.90-0.95)**:
  - ✅ More KV cache capacity
  - ✅ Can handle longer sequences
  - ✅ Better throughput with concurrent requests
  - ❌ Risk of OOM if model activations need more space
- **Lower values (0.75-0.85)**:
  - ✅ More headroom for model activations
  - ✅ Safer for complex generation tasks
  - ❌ Less KV cache capacity
  - ❌ May limit concurrent requests

**Trade-off**: Balance between KV cache size and activation memory

### 3. Max Total Tokens (`--max-total-tokens`)

**Description**: Maximum number of tokens (input + output) across all requests

**Parameter**: `max_total_tokens`

**Values**:
- Type: Integer
- Range: Model-dependent
- Default: Calculated from `mem_frac`

**Impact**:
- Controls total KV cache capacity
- Directly affects maximum concurrency
- Formula: `max_concurrent_requests = max_total_tokens / average_sequence_length`

### 4. Schedule Policy (`--schedule-policy`)

**Description**: Request scheduling algorithm

**Parameter**: `schedule_policy`

**Values**:
- `lpm` (Least Pending Memory): Default, prioritizes requests with less pending work
- `fcfs` (First-Come First-Serve): FIFO scheduling
- `sjf` (Shortest Job First): Prioritizes shorter sequences

**Impact on Metrics**:
- `lpm`: Best for mixed workloads
- `fcfs`: Predictable latency, simple
- `sjf`: Minimizes average latency for varied lengths

---

## Performance Metrics

The autotuner tracks these metrics to evaluate configurations:

### Primary Metrics

#### 1. End-to-End Latency
**Description**: Total time from request submission to completion

**Measurement**:
- **Unit**: Milliseconds (ms) or seconds (s)
- **Calculation**: `completion_time - request_time`
- **Variants**:
  - Mean latency: Average across all requests
  - P50 (median): 50th percentile
  - P95: 95th percentile (tail latency)
  - P99: 99th percentile (worst-case latency)

**Optimization Goal**: Minimize (especially P95/P99 for consistent UX)

**Affected By**:
- `mem_frac`: Affects available KV cache
- `schedule_policy`: Affects request prioritization
- Batch size and concurrency

#### 2. Time to First Token (TTFT)
**Description**: Time from request to first generated token

**Measurement**:
- **Unit**: Milliseconds (ms)
- **Calculation**: `first_token_time - request_time`

**Optimization Goal**: Minimize (critical for perceived responsiveness)

**Affected By**:
- Prompt length
- KV cache pressure
- Scheduling policy
- GPU utilization

#### 3. Tokens Per Second (Throughput)
**Description**: Rate of token generation

**Measurement**:
- **Unit**: Tokens/second
- **Variants**:
  - Per-request throughput: `output_tokens / generation_time`
  - System throughput: `total_tokens / total_time`

**Optimization Goal**: Maximize

**Affected By**:
- `tp_size`: More GPUs → higher throughput (with caveats)
- `mem_frac`: More cache → more concurrent requests → higher aggregate throughput
- Batch size

#### 4. Time Per Output Token (TPOT)
**Description**: Average time to generate each output token after the first

**Measurement**:
- **Unit**: Milliseconds per token (ms/token)
- **Calculation**: `(completion_time - first_token_time) / (num_output_tokens - 1)`

**Optimization Goal**: Minimize

**Affected By**:
- Model size
- KV cache efficiency
- Decode batch size

### Secondary Metrics

#### 5. GPU Memory Utilization
**Description**: Percentage of GPU memory in use

**Measurement**:
- **Unit**: Percentage (%) or bytes
- **Source**: `nvidia-smi` or CUDA runtime

**Optimization Goal**: High utilization without OOM

**Affected By**:
- `mem_frac`: Directly controls allocation
- `max_total_tokens`: Affects KV cache size
- Active batch size

#### 6. Request Success Rate
**Description**: Percentage of requests completed successfully

**Measurement**:
- **Unit**: Percentage (%)
- **Calculation**: `successful_requests / total_requests * 100`

**Optimization Goal**: 100%

**Failure Modes**:
- OOM errors → reduce `mem_frac` or `max_total_tokens`
- Timeout errors → increase timeouts or reduce load
- API errors → check configuration

#### 7. Concurrent Request Capacity
**Description**: Maximum number of requests handled simultaneously

**Measurement**:
- **Unit**: Integer count
- **Method**: Gradually increase load until saturation

**Optimization Goal**: Maximize while maintaining SLAs

**Affected By**:
- `mem_frac`: More cache → more concurrency
- `max_total_tokens`: Hard limit on total tokens
- Average sequence length

---

## Parameter Tuning Guide

### Tuning for Different Objectives

#### Objective 1: Minimize Latency (Interactive Use Cases)

**Goal**: Lowest P95 latency for single-user experience

**Recommended Settings**:
```json
{
  "tp_size": 1,
  "mem_frac": 0.85,
  "schedule_policy": "fcfs"
}
```

**Rationale**:
- Lower `mem_frac` provides headroom for consistent performance
- `tp_size=1` minimizes communication overhead
- FCFS ensures predictable ordering

#### Objective 2: Maximize Throughput (Batch Processing)

**Goal**: Highest tokens/second for batch workloads

**Recommended Settings**:
```json
{
  "tp_size": 2-4,
  "mem_frac": 0.92,
  "schedule_policy": "lpm"
}
```

**Rationale**:
- Higher `mem_frac` allows more concurrent requests
- Tensor parallelism helps with large batches
- LPM optimizes for memory efficiency

#### Objective 3: Balance (Production API)

**Goal**: Good throughput with acceptable P95 latency

**Recommended Settings**:
```json
{
  "tp_size": 1-2,
  "mem_frac": 0.88,
  "schedule_policy": "lpm"
}
```

**Rationale**:
- Balanced memory allocation
- Moderate parallelism
- Adaptive scheduling

### Tuning Process

**1. Establish Baseline**
```json
{
  "tp_size": 1,
  "mem_frac": 0.85
}
```

**2. Tune Memory Fraction**
- Increase `mem_frac` until OOM or diminishing returns
- Monitor: Memory utilization, success rate

**3. Tune Tensor Parallelism**
- Increase `tp_size` if model doesn't fit
- Compare throughput at different values
- Monitor: Communication overhead, GPU utilization

**4. Tune Scheduling**
- Test different policies under realistic load
- Monitor: P95 latency, throughput variance

**5. Validate**
- Run extended tests with production-like traffic
- Monitor for stability and OOM events

---

## Metrics Collection

### Collection Methods

#### 1. Direct Metrics (from genai-bench)

genai-bench provides these metrics directly:

```json
{
  "end_to_end_latency_s": {
    "mean": 1.234,
    "median": 1.123,
    "p95": 2.345,
    "p99": 3.456
  },
  "ttft_s": {
    "mean": 0.156,
    "median": 0.145,
    "p95": 0.234
  },
  "throughput_token_per_s": {
    "mean": 45.6
  },
  "tpot_s": {
    "mean": 0.023,
    "median": 0.022
  }
}
```

#### 2. SGLang Server Metrics

SGLang exposes Prometheus metrics at `/metrics`:

**Request Metrics**:
- `sglang_requests_total`: Total requests processed
- `sglang_requests_success`: Successful completions
- `sglang_requests_failed`: Failed requests

**Latency Metrics**:
- `sglang_request_latency_seconds`: Request latency histogram
- `sglang_ttft_seconds`: Time to first token histogram
- `sglang_tpot_seconds`: Time per output token histogram

**Resource Metrics**:
- `sglang_gpu_memory_used_bytes`: GPU memory usage
- `sglang_kv_cache_used_tokens`: KV cache utilization
- `sglang_batch_size`: Current decode batch size

#### 3. System Metrics (via nvidia-smi)

```bash
# GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
```

### Metric Aggregation

The autotuner collects and aggregates:

1. **Per-Run Metrics**: From each benchmark execution
2. **Aggregated Metrics**: Across multiple runs for robustness
3. **Objective Score**: Computed based on optimization goal

**Example Objective Functions**:

```python
# Minimize latency
score = metrics['end_to_end_latency_s']['p95']

# Maximize throughput
score = -metrics['throughput_token_per_s']['mean']  # Negative for minimization

# Balance latency and throughput
score = metrics['end_to_end_latency_s']['p95'] / metrics['throughput_token_per_s']['mean']
```

---

## Appendix: Parameter Summary Table

| Parameter | Flag | Type | Range | Default | Primary Impact |
|-----------|------|------|-------|---------|----------------|
| Tensor Parallelism | `--tp-size` | int | 1-8 | 1 | Model capacity, throughput |
| Memory Fraction | `--mem-frac` | float | 0.0-1.0 | 0.9 | Concurrency, stability |
| Max Total Tokens | `--max-total-tokens` | int | varies | auto | KV cache capacity |
| Schedule Policy | `--schedule-policy` | string | lpm/fcfs/sjf | lpm | Request ordering |
| Max Prefill Tokens | `--max-prefill-tokens` | int | varies | 16384 | Long prompt handling |
| Chunked Prefill | `--chunked-prefill-size` | int | varies | 8192 | Prompt processing |

---

## References

- **SGLang Documentation**: https://github.com/sgl-project/sglang
- **genai-bench Documentation**: https://github.com/sgl-project/genai-bench
- **OME Documentation**: https://github.com/sgl-project/ome
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**Last Updated**: 2025-10-22
**Autotuner Version**: Prototype v0.1.0
