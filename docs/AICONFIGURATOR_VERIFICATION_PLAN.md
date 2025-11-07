# Aiconfigurator Verification Plan: Data Alignment & Comparison Strategy

## Executive Summary

**Goal**: Run actual benchmark experiments to verify aiconfigurator's static performance estimates, enabling users to compare predicted vs. actual metrics.

**Challenge**: Align data dimensions between aiconfigurator's prediction system and inference-autotuner's measurement system.

---

## 1. Understanding the Two Systems

### Aiconfigurator (Static Estimation)
**Input:**
- Model specification (LLAMA, GPT, MoE with layer counts, dimensions)
- Hardware specification (h100_sxm, h200_sxm, etc.)
- Configuration space (TP, PP, DP, EP, quantization modes)
- Workload (isl, osl, batch sizes)
- SLA constraints (TTFT, TPOT thresholds)

**Process:**
- Operation-level performance modeling (GEMM, Attention, AllReduce)
- Query pre-collected CSV performance databases
- Interpolation for unseen configurations
- Pareto frontier analysis

**Output:**
- Predicted throughput (tokens/s/gpu)
- Predicted TTFT (ms)
- Predicted TPOT (ms)
- Predicted latency percentiles (P50, P90, P99)
- Pareto-optimal configurations
- **No actual deployment or benchmarking**

---

### Inference-Autotuner (Actual Measurement)
**Input:**
- Task JSON with model path, runtime, parameters grid
- Deployment mode (Docker/OME)
- Benchmark configuration (traffic scenarios, concurrency)
- SLO constraints

**Process:**
- Deploy inference service (Docker container or K8s)
- Run genai-bench against deployed service
- Parse metrics from benchmark results
- Calculate objective scores with SLO penalties

**Output:**
- Actual throughput (requests/s, tokens/s)
- Actual TTFT (mean, median, P90, P99)
- Actual TPOT (mean, median, P90, P99)
- Actual latency (mean, median, P90, P99)
- Best configuration based on objective

---

## 2. Critical Data Dimension Alignment Issues

### 2.1 Configuration Parameters

| Dimension | Aiconfigurator | Inference-Autotuner | Mapping Required |
|-----------|----------------|---------------------|------------------|
| **Tensor Parallel** | `tp_list: [1,2,4,8]` | `tp-size: [1,2,4,8]` (SGLang)<br>`tensor-parallel-size: [1,2,4,8]` (vLLM) | ✅ Direct mapping |
| **Pipeline Parallel** | `pp_list: [1,2,4]` | `pipeline-parallel-size: [1,2,4]` (vLLM) | ✅ Direct mapping<br>⚠️ SGLang doesn't support PP |
| **Data Parallel** | `dp_list: [1,2,4]` | `data-parallel-size: [1,2,4]` (vLLM) | ✅ Direct mapping<br>⚠️ SGLang uses replicas differently |
| **Expert Parallel** | `moe_ep_list: [1,2,4,8]` | `expert-parallel-size: [1,2,4,8]` (vLLM) | ✅ Direct mapping<br>⚠️ SGLang MoE support varies |
| **Quantization** | Per-component:<br>`gemm_quant_mode: "fp8_block"`<br>`kvcache_quant_mode: "fp8"`<br>`fmha_quant_mode: "fp8"` | Mixed:<br>`quantization: "fp8"` (general)<br>`kv-cache-dtype: "fp8"` (specific) | ⚠️ **Complex mapping needed**<br>SGLang/vLLM don't expose all components |
| **Memory** | Derived from config | `mem-fraction-static: [0.7,0.8,0.9]` (SGLang)<br>`gpu-memory-utilization: [0.85,0.9]` (vLLM) | ⚠️ Different memory models |
| **Scheduling** | Not in aiconfigurator | `schedule-policy: ["lpm", "fcfs"]` (SGLang) | ❌ No equivalent in aiconfigurator |

**Key Issue:** Aiconfigurator uses fine-grained per-component quantization, but SGLang/vLLM expose coarser controls.

---

### 2.2 Model Specification

| Dimension | Aiconfigurator | Inference-Autotuner | Mapping Required |
|-----------|----------------|---------------------|------------------|
| **Model Type** | Enum: `LLAMA_7B`, `LLAMA_70B`, `QWEN3_32B`, etc. | String path: `/mnt/data/models/llama-3-2-1b-instruct` | ⚠️ Need model registry mapping |
| **Architecture** | Structured:<br>`layers=32, hidden=4096, inter=11008, heads=32` | Inferred from model files | ⚠️ Need to extract model config |
| **MoE Info** | `num_experts=8, active_experts=2` | Inferred from model | ⚠️ Need MoE detection |
| **Context Length** | Explicit: `max_position_embeddings=8192` | CLI param: `context-length=8192` | ✅ Direct mapping |

**Key Issue:** Aiconfigurator needs structured model info, but we only have model paths. Need model introspection.

---

### 2.3 Workload Specification

| Dimension | Aiconfigurator | Inference-Autotuner | Mapping Required |
|-----------|----------------|---------------------|------------------|
| **Input Length** | `isl: 4000` (tokens) | `traffic_scenarios: ["D(4000,1000)"]` | ✅ Extract from traffic pattern |
| **Output Length** | `osl: 1000` (tokens) | `traffic_scenarios: ["D(4000,1000)"]` | ✅ Extract from traffic pattern |
| **Batch Size** | `context_batch_size: 32`<br>`generation_batch_size: 512` | Controlled by runtime (continuous batching) | ⚠️ Not directly controllable |
| **Concurrency** | Derived | `num_concurrency: [1,4,8]` | ⚠️ Different batching models |
| **Request Rate** | Static analysis (no rate) | `traffic_scenarios: ["D(100,100)"]` = Poisson arrival | ❌ Fundamentally different |

**Key Issue:** Aiconfigurator does static analysis (no time dimension), while genai-bench simulates request arrivals over time.

---

### 2.4 Hardware Specification

| Dimension | Aiconfigurator | Inference-Autotuner | Mapping Required |
|-----------|----------------|---------------------|------------------|
| **GPU Type** | Enum: `h100_sxm`, `h200_sxm`, `a100_sxm` | Inferred from `nvidia-smi` | ⚠️ Need GPU detection |
| **Num GPUs** | `total_gpus: 32` | Specified in task or auto-detect | ✅ Direct mapping |
| **Interconnect** | Implicit in system type | Not modeled | ❌ Can't control |
| **Memory per GPU** | System-specific (80GB for H100) | Auto-detected | ✅ Can query |

**Key Issue:** Need to detect GPU type and map to aiconfigurator's system types.

---

### 2.5 Metrics Output

| Metric | Aiconfigurator Output | Genai-Bench Output | Alignment Needed |
|--------|----------------------|-------------------|------------------|
| **Throughput** | `throughput: 913.82` (tokens/s/gpu) | `output_throughput: 3655.28` (tokens/s total) | ⚠️ Need to divide by num_gpus |
| **TTFT** | `ttft: 202.65` (ms, estimated) | `ttft_mean: 205.3, ttft_p90: 220.1` (ms) | ✅ Compare estimated vs mean |
| **TPOT** | `tpot: 8.07` (ms, estimated) | `tpot_mean: 8.5, tpot_p90: 9.2` (ms) | ✅ Compare estimated vs mean |
| **Latency** | `latency_p90: 4.2` (s, estimated) | `e2e_latency_p90: 4.5` (s) | ✅ Compare percentiles |
| **User Throughput** | `user_throughput: 123.92` (tokens/s/user) | `request_throughput: 10.5` (req/s) | ⚠️ Different units |
| **Deployment** | `config: {tp=4, pp=1, batch=32}` | `parameters: {"tp-size": 4}` | ✅ Record for comparison |

**Key Issue:** Units differ (tokens/s/gpu vs tokens/s total, requests/s vs tokens/s/user).

---

## 3. Proposed Verification Workflow

### Phase 1: Configuration Translation
```
Aiconfigurator Config → Inference-Autotuner Task
```

**Input:** Aiconfigurator's Pareto-optimal configuration
```yaml
# Aiconfigurator output
model_name: "QWEN3_32B"
serving_mode: "disagg"
system_name: "h200_sxm"
total_gpus: 32

prefill_worker_config:
  tp: 4
  pp: 1
  num_gpu_per_worker: 4
  gemm_quant_mode: "fp8_block"
  kvcache_quant_mode: "fp8"
  fmha_quant_mode: "fp8"

decode_worker_config:
  tp: 4
  pp: 1
  num_gpu_per_worker: 4

replica_config:
  num_replicas: 4

estimated_metrics:
  throughput: 913.82  # tokens/s/gpu
  ttft: 202.65  # ms
  tpot: 8.07  # ms
```

**Output:** Inference-Autotuner task JSON
```json
{
  "task_name": "verify_aiconfigurator_disagg_qwen32b",
  "model": {
    "id_or_path": "qwen-3-32b-instruct",
    "namespace": "verification"
  },
  "base_runtime": "sglang",
  "runtime_image_tag": "v0.5.2-cu126",

  "parameters": {
    "tp-size": [4],
    "quantization": ["fp8"],
    "kv-cache-dtype": ["fp8"],
    "mem-fraction-static": [0.85]
  },

  "benchmark": {
    "task": "text-to-text",
    "traffic_scenarios": ["D(4000,1000)"],
    "num_concurrency": [8]
  },

  "metadata": {
    "aiconfigurator_config": {...},
    "aiconfigurator_estimates": {
      "throughput_tokens_per_s_per_gpu": 913.82,
      "ttft_ms": 202.65,
      "tpot_ms": 8.07,
      "latency_p90_s": 4.2
    }
  }
}
```

---

### Phase 2: Experiment Execution
```
Run Task → Measure Actual Metrics → Store Results
```

**Process:**
1. Deploy inference service with aiconfigurator's recommended config
2. Run genai-bench with matching workload (isl=4000, osl=1000)
3. Parse metrics from benchmark results
4. Normalize metrics to match aiconfigurator's units

**Example Normalization:**
```python
# Genai-bench output
genai_bench_results = {
    "output_throughput": 29241.28,  # tokens/s total
    "ttft_mean": 205.3,  # ms
    "tpot_mean": 8.5,  # ms
    "e2e_latency_p90": 4.5  # s
}

# Normalize to aiconfigurator units
num_gpus = 32
normalized_results = {
    "throughput_tokens_per_s_per_gpu": 29241.28 / 32,  # = 913.79
    "ttft_ms": 205.3,
    "tpot_ms": 8.5,
    "latency_p90_s": 4.5
}
```

---

### Phase 3: Comparison & Visualization
```
Actual Metrics vs Predicted Metrics → Comparison Report
```

**Comparison Metrics:**
```python
comparison = {
    "throughput": {
        "predicted": 913.82,
        "actual": 913.79,
        "error_pct": 0.003,  # 0.3% error
        "status": "excellent"
    },
    "ttft": {
        "predicted": 202.65,
        "actual": 205.3,
        "error_pct": 1.31,  # 1.31% error
        "status": "good"
    },
    "tpot": {
        "predicted": 8.07,
        "actual": 8.5,
        "error_pct": 5.33,  # 5.33% error
        "status": "acceptable"
    },
    "latency_p90": {
        "predicted": 4.2,
        "actual": 4.5,
        "error_pct": 7.14,  # 7.14% error
        "status": "acceptable"
    }
}
```

**Visualization:**
- Side-by-side bar charts (predicted vs actual)
- Error percentage plots
- Scatter plots (predicted vs actual with diagonal)
- Pareto frontier overlay (predicted vs actual points)

---

## 4. Implementation Strategy

### 4.1 Phase 1: Model Registry & Introspection
**File:** `src/utils/model_registry.py`

**Purpose:** Map model paths to aiconfigurator model specifications

```python
class ModelRegistry:
    """Registry mapping model paths to aiconfigurator model specs."""

    KNOWN_MODELS = {
        "llama-3-2-1b-instruct": {
            "aiconfigurator_name": "LLAMA_1B",
            "layers": 16,
            "hidden": 2048,
            "inter": 8192,
            "heads": 32,
            "kv_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "architecture": "llama",
            "is_moe": False
        },
        "qwen-3-32b-instruct": {
            "aiconfigurator_name": "QWEN3_32B",
            "layers": 64,
            "hidden": 5120,
            "inter": 27648,
            "heads": 40,
            "kv_heads": 8,
            "architecture": "qwen",
            "is_moe": False
        },
        # ... more models
    }

    @classmethod
    def get_model_spec(cls, model_path: str) -> ModelSpec:
        """Get model specification for aiconfigurator."""
        ...

    @classmethod
    def introspect_model(cls, model_path: str) -> ModelSpec:
        """Load config.json and extract model architecture."""
        ...
```

---

### 4.2 Phase 2: Configuration Translator
**File:** `src/utils/aiconfigurator_translator.py`

**Purpose:** Translate between aiconfigurator configs and inference-autotuner tasks

```python
class AiconfiguratorTranslator:
    """Bidirectional translation between aiconfigurator and inference-autotuner."""

    @staticmethod
    def aiconfig_to_task(
        aiconfig: Dict[str, Any],
        model_path: str,
        runtime: str = "sglang"
    ) -> Dict[str, Any]:
        """
        Convert aiconfigurator configuration to inference-autotuner task.

        Args:
            aiconfig: Aiconfigurator configuration (worker_config, etc.)
            model_path: Model path in /mnt/data/models/
            runtime: Target runtime (sglang or vllm)

        Returns:
            Task JSON for inference-autotuner
        """
        ...

    @staticmethod
    def task_to_aiconfig(
        task: Dict[str, Any],
        model_spec: ModelSpec
    ) -> Dict[str, Any]:
        """
        Convert inference-autotuner task to aiconfigurator input.

        Args:
            task: Inference-autotuner task JSON
            model_spec: Model architecture specification

        Returns:
            Aiconfigurator configuration
        """
        ...

    @staticmethod
    def normalize_metrics(
        genai_bench_results: Dict[str, Any],
        num_gpus: int
    ) -> Dict[str, float]:
        """
        Normalize genai-bench metrics to aiconfigurator units.

        Returns:
            Metrics in aiconfigurator format (tokens/s/gpu, ms, etc.)
        """
        ...
```

---

### 4.3 Phase 3: Verification Experiment Runner
**File:** `src/verification/experiment_runner.py`

**Purpose:** Run verification experiments comparing aiconfigurator predictions with actual results

```python
class VerificationExperiment:
    """Run verification experiments for aiconfigurator predictions."""

    def __init__(
        self,
        aiconfigurator_config: Dict[str, Any],
        model_path: str,
        runtime: str,
        deployment_mode: str = "docker"
    ):
        self.aiconfig = aiconfigurator_config
        self.model_path = model_path
        self.runtime = runtime
        self.deployment_mode = deployment_mode

    async def run(self) -> VerificationResult:
        """
        Run verification experiment.

        Steps:
        1. Translate aiconfigurator config to task
        2. Create and start task
        3. Wait for completion
        4. Normalize metrics
        5. Compare with predictions
        6. Generate report
        """
        ...

    def compare_metrics(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float]
    ) -> ComparisonReport:
        """Compare predicted vs actual metrics."""
        ...

    def generate_report(self) -> str:
        """Generate markdown comparison report."""
        ...
```

---

### 4.4 Phase 4: Comparison Visualization
**File:** `frontend/src/components/AiconfiguratorComparison.tsx`

**Purpose:** Visualize aiconfigurator predictions vs actual measurements

**Features:**
- Side-by-side metric comparison
- Error percentage charts
- Scatter plots with prediction accuracy
- Configuration diff viewer
- Export comparison data

---

## 5. Critical Alignment Challenges

### Challenge 1: Quantization Granularity Mismatch
**Problem:** Aiconfigurator controls gemm_quant_mode, kvcache_quant_mode, fmha_quant_mode separately. SGLang/vLLM have coarser controls.

**Solutions:**
- Option A: Use closest approximation (e.g., aiconfigurator's "fp8_block" → SGLang's "fp8")
- Option B: Document limitations and only test configurations with supported quantization
- Option C: Extend SGLang/vLLM with fine-grained quantization controls (long-term)

**Recommendation:** Option B for now, document discrepancies in comparison report.

---

### Challenge 2: Disaggregated Serving Mode
**Problem:** Aiconfigurator's key innovation (separate prefill/decode workers) is not implemented in current inference-autotuner.

**Solutions:**
- Option A: Only test aggregated (agg) mode for now
- Option B: Implement disaggregated serving in Docker mode (2-4 weeks work)
- Option C: Test prefill/decode phases separately (manual orchestration)

**Recommendation:** Option A initially (test agg mode), then Option B after config refactor complete.

---

### Challenge 3: Workload Model Differences
**Problem:** Aiconfigurator does static analysis (single batch), genai-bench does dynamic simulation (request arrivals over time).

**Solutions:**
- Option A: Use low concurrency (num_concurrency=1) to approximate single-batch behavior
- Option B: Use aiconfigurator's batch_size to guide genai-bench concurrency selection
- Option C: Run genai-bench in "static batch" mode if available

**Recommendation:** Option A, document that we're comparing "static capacity" vs "dynamic throughput under load".

---

### Challenge 4: Data Parallel and Replicas
**Problem:** Aiconfigurator uses replica-based scaling (4 replicas × 8 GPUs each). Inference-autotuner doesn't have replica concept.

**Solutions:**
- Option A: Treat TP × PP as "GPU per worker", ignore replication
- Option B: Launch multiple Docker containers to simulate replicas
- Option C: Focus on per-replica metrics (divide by num_replicas)

**Recommendation:** Option A for simplicity, document that we test single-replica performance.

---

## 6. Verification Experiment Plan

### Experiment 1: Simple Baseline
**Goal:** Verify basic configuration translation and metric collection

**Setup:**
- Model: llama-3-2-1b-instruct
- Config: TP=1, FP16, no quantization
- Workload: D(100, 100) - short sequences
- Hardware: Single GPU (H100)

**Expected:**
- Metric collection works
- Units aligned
- Basic comparison report generated

---

### Experiment 2: Quantization Comparison
**Goal:** Test quantization mode mapping

**Setup:**
- Model: llama-3-2-1b-instruct
- Configs: [FP16, FP8] × [TP=1, TP=2]
- Workload: D(1000, 500)
- Hardware: 2 GPUs

**Expected:**
- FP8 shows ~1.5-2x throughput improvement
- Aiconfigurator predictions match actual within 10-15%

---

### Experiment 3: Parallelism Scaling
**Goal:** Verify tensor parallel scaling predictions

**Setup:**
- Model: qwen-3-32b-instruct (larger model)
- Configs: TP=[1, 2, 4, 8]
- Workload: D(4000, 1000) - long sequences
- Hardware: 8 GPUs

**Expected:**
- Linear scaling for TP (with communication overhead)
- Aiconfigurator predicts communication costs accurately

---

### Experiment 4: MoE Model (if supported)
**Goal:** Test MoE-specific configurations

**Setup:**
- Model: Mixtral-8x7B or Qwen MoE
- Configs: EP=[1, 2, 4, 8]
- Workload: D(2000, 500)
- Hardware: 8 GPUs

**Expected:**
- Expert parallel scaling behavior
- MoE-specific metric alignment

---

## 7. Success Metrics

### Metric Accuracy Targets
- **Throughput:** Predicted within 10% of actual
- **TTFT:** Predicted within 15% of actual (more variance expected)
- **TPOT:** Predicted within 10% of actual
- **Latency P90:** Predicted within 20% of actual

### Coverage Targets
- Test at least 10 configurations per model
- Test 3+ different models (dense models)
- Test 2+ GPU counts (1, 2, 4, 8)
- Test 2+ quantization modes (FP16, FP8)

### Deliverables
- [ ] Model registry with 5+ models mapped
- [ ] Configuration translator (bidirectional)
- [ ] Metric normalization utilities
- [ ] Verification experiment runner
- [ ] Comparison visualization UI
- [ ] Verification report for 10+ experiments
- [ ] Documentation of alignment challenges and solutions

---

## 8. Timeline

### Week 1-2: Foundation
- Build model registry and introspection
- Implement configuration translator
- Create metric normalization utilities
- Document mapping rules

### Week 3-4: Experiment Infrastructure
- Build verification experiment runner
- API endpoints for verification tasks
- Database schema for storing predictions + actuals
- Basic comparison report generation

### Week 5-6: UI & Visualization
- Frontend comparison visualization
- Interactive prediction vs actual charts
- Configuration diff viewer
- Export functionality

### Week 7-8: Validation Experiments
- Run 10+ verification experiments
- Analyze prediction accuracy
- Document discrepancies
- Generate validation report
- Publish findings

---

## 9. Open Questions

1. **Which runtime to prioritize?** SGLang or vLLM? (SGLang recommended due to better alignment)
2. **Disagg mode timing?** Wait for full implementation or test manually?
3. **Aiconfigurator version?** Use submodule version or need to sync with upstream?
4. **GPU access?** Do we have access to H100/H200 for realistic tests?
5. **Model availability?** Which models do we have in /mnt/data/models/?
6. **Workload patterns?** Should we test beyond D(isl, osl) patterns?
7. **Reporting format?** Markdown reports, JSON data, or interactive dashboard?

---

## 10. Next Steps (User Decision Required)

Before proceeding with grouped config refactor, we need to decide:

**Option A: Config Refactor First, Verification Later**
- Complete grouped config system (8 weeks)
- Then build verification infrastructure (8 weeks)
- Total: 16 weeks

**Option B: Verification First, Refactor Later**
- Build verification system with current flat config (4 weeks)
- Validate aiconfigurator accuracy
- Then refactor config system if needed (8 weeks)
- Total: 12 weeks, but verification results inform refactor

**Option C: Hybrid Approach**
- Build minimal config alignment (2 weeks)
- Build verification infrastructure (4 weeks)
- Run initial experiments (2 weeks)
- Then decide on full refactor based on learnings
- Total: 8 weeks to first results

**Recommendation:** Option C (Hybrid) - Get verification results first to inform whether full refactor is necessary.

---

## Questions for User

1. What's the priority: Config system refactor or aiconfigurator verification?
2. Which runtime should we focus on first (SGLang or vLLM)?
3. Do we have access to H100/H200 GPUs for realistic tests?
4. Should we implement disaggregated serving mode before verification?
5. What accuracy threshold would be "good enough" (10%? 15%? 20%)?
