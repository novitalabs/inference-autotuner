# aiconfigurator: Key Features for Inference-Autotuner Integration

## Executive Summary

**aiconfigurator** is NVIDIA's AI system for automatically optimizing LLM inference deployments. After a comprehensive deep dive, we've identified **10 high-value features** that can significantly enhance the inference-autotuner project.

**Core Innovation**: Disaggregated serving (separating prefill/decode phases) achieves **1.7x-2x performance improvements** under SLA constraints.

---

## Top 10 Features to Adapt

### 1. **Layered Configuration Factory** ⭐⭐⭐⭐⭐

**What it does**: Composes configurations from multiple layers (base → mode → profile → user overrides)

**Current aiconfigurator implementation**:
```python
@dataclass(frozen=True)
class ConfigLayer:
    name: str
    data: dict | Callable[[Context], dict]
    condition: Callable[[Context], bool] | None = None

    def applies_to(self, ctx: Context) -> bool:
        return self.condition is None or self.condition(ctx)
```

**How to adapt for inference-autotuner**:
- Build hierarchical SLO configurations: Base SLO → Model-specific → Task-specific
- Enable conditional layer application based on runtime/model type
- Support YAML patch/replace modes for flexible overrides

**Benefits**:
- Clean separation of concerns
- Easy to extend with new profiles
- Declarative configuration inheritance

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/task.py`

---

### 2. **Pareto Frontier Analysis** ⭐⭐⭐⭐⭐

**What it does**: Multi-objective optimization finding trade-offs between competing objectives

**Current aiconfigurator implementation**:
```python
def find_pareto_frontier(results):
    # For each configuration:
    #   1. Compute multi-objective score
    #   2. Check SLA constraints
    #   3. Apply exponential penalties for violations
    #   4. Return non-dominated configurations

    pareto_set = {configs where no other is strictly better in all objectives}
```

**How to adapt for inference-autotuner**:
- Enable multi-objective optimization: throughput vs latency vs cost
- Find optimal parameter configurations that aren't strictly dominated
- Visualize trade-off curves in the frontend

**Benefits**:
- Discover non-obvious optimal configurations
- Present users with multiple valid choices
- Handle conflicting objectives systematically

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/pareto_analysis.py`

---

### 3. **Exponential Penalty Functions for SLA Violations** ⭐⭐⭐⭐⭐

**What it does**: Graceful scoring degradation for constraint violations (not just pass/fail)

**Current aiconfigurator implementation**:
```python
penalty = weight × exp(violation_ratio / steepness)
violation_ratio = (actual - threshold) / threshold

# Soft constraint (hard_fail=false):
score *= (1 - penalty)

# Hard constraint (hard_fail=true):
if violation_ratio > fail_ratio:
    mark_experiment_as_FAILED()
```

**How to adapt for inference-autotuner**:
- **Already implemented!** Your current SLO scoring in `src/orchestrator.py` uses this pattern
- Can enhance with tunable steepness parameter
- Add tiered penalties (warning/error thresholds)

**Benefits**:
- More nuanced scoring than binary pass/fail
- Captures "how badly" a constraint is violated
- Enables soft vs hard constraint distinction

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/pareto_analysis.py`

---

### 4. **Backend Strategy Pattern** ⭐⭐⭐⭐

**What it does**: Abstract interface for different inference engines with unified API

**Current aiconfigurator implementation**:
```python
class BaseBackend(ABC):
    @abstractmethod
    def run_static(...) -> InferenceSummary: ...

    @abstractmethod
    def run_agg(...) -> InferenceSummary: ...

    @abstractmethod
    def find_best_result_under_constraints(...): ...

# Factory pattern
backend = get_backend(backend_name)  # Returns TRTLLM/SGLang/VLLM
```

**How to adapt for inference-autotuner**:
- **Already have similar pattern!** Your `BaseModelController` implements this
- Can enhance by extracting backend-specific logic from controllers
- Add performance prediction methods to backend interface

**Benefits**:
- Easy to add new inference engines
- Clean separation of framework-specific logic
- Unified benchmarking interface

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/backends/`

---

### 5. **Operation-Based Performance Modeling** ⭐⭐⭐⭐

**What it does**: Breaks inference into composable operations for performance prediction

**Current aiconfigurator implementation**:
```python
# Operations modeled:
context_ops = [GEMM, Attention, Embedding, AllReduce, KVCache, ...]
generation_ops = [GEMM, Attention, Sampling, P2P, ...]

# Query pre-collected CSV database
latency = db.query_gemm(quant_mode, tp_size, m, n, k)
latency += db.query_attention(quant_mode, batch_size, seq_len)
# Interpolate for unseen configurations
```

**How to adapt for inference-autotuner**:
- Build lightweight performance database from benchmark results
- Enable "what-if" analysis without running full experiments
- Predict performance for unseen parameter combinations

**Benefits**:
- Faster parameter space exploration
- Reduced benchmarking cost
- Performance prediction before deployment

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/{operations.py,perf_database.py}`

---

### 6. **Configuration Profiles & Presets** ⭐⭐⭐⭐

**What it does**: Pre-built configuration templates for common scenarios

**Current aiconfigurator implementation**:
```yaml
profiles:
  - fp8_default:      # FP8 quantization for all operations
      gemm_quant_mode: "fp8_block"
      kvcache_quant_mode: "fp8"
      fmha_quant_mode: "fp8"

  - float16_default:  # No quantization
      gemm_quant_mode: "float16"
      kvcache_quant_mode: "float16"
```

**How to adapt for inference-autotuner**:
- **Already have parameter presets in frontend!** Can enhance with:
  - SLO presets (strict_latency, high_throughput, balanced)
  - Model-specific parameter presets
  - Hardware-optimized presets (A100, H100, consumer GPUs)

**Benefits**:
- Lower barrier to entry for users
- Encode best practices
- Quick starting points for experimentation

**Code location**: `aiconfigurator/src/aiconfigurator/sdk/task.py` (profiles system)

---

### 7. **Heterogeneous Deployment Support** ⭐⭐⭐

**What it does**: Different hardware for different phases (prefill on H200, decode on H100)

**Current aiconfigurator implementation**:
```yaml
system_name: "h200_sxm"         # Prefill hardware
decode_system_name: "h100_sxm"  # Decode hardware (cheaper)
```

**How to adapt for inference-autotuner**:
- Enable cost optimization by using different GPU types
- Support multi-node deployments with heterogeneous hardware
- Add cost tracking to objective scoring

**Benefits**:
- Significant cost savings (use cheaper GPUs for less intensive phases)
- Better resource utilization
- More deployment flexibility

**Implementation effort**: Medium (requires multi-hardware controller support)

---

### 8. **Multi-Experiment Comparison** ⭐⭐⭐

**What it does**: Run multiple experiments in parallel and compare results

**Current aiconfigurator implementation**:
```yaml
exps:
  - exp_disagg_h200
  - exp_agg_h100
  - exp_hybrid

# Generates comparison tables and pareto plots
```

**How to adapt for inference-autotuner**:
- **Partially implemented** (can run experiments sequentially within a task)
- Add parallel task execution
- Enhance frontend with cross-task comparison view
- Generate comparison reports automatically

**Benefits**:
- Faster iteration cycles
- Easy A/B testing of configurations
- Better visualization of trade-offs

**Frontend work required**: Add comparison dashboard

---

### 9. **Replica-Based Scaling Model** ⭐⭐⭐

**What it does**: Treats deployments as replicated units for predictable scaling

**Current aiconfigurator implementation**:
```python
# Example: 32 GPUs total
replica = 4 prefill workers (1 GPU each) + 1 decode worker (4 GPUs)
total_replicas = 32 / (4 + 4) = 4 replicas

# Scale throughput linearly: base_throughput × num_replicas
```

**How to adapt for inference-autotuner**:
- Model horizontal scaling behavior
- Predict performance at different scales
- Enable cost-performance optimization

**Benefits**:
- Predictable scaling characteristics
- Easier capacity planning
- Cost modeling for different scale scenarios

**Implementation effort**: Medium (requires scaling simulation logic)

---

### 10. **Template-Based Configuration Generation** ⭐⭐⭐

**What it does**: Generate framework-specific configs from templates

**Current aiconfigurator implementation**:
```python
# Jinja2 templates for:
# - Framework configs (TRTLLM, SGLang, VLLM)
# - Kubernetes deployment manifests
# - Shell deployment scripts

artifacts = generate_backend_config(
    cfg=config,
    backend="trtllm",
    version="1.0.0rc3",
    save_dir="/tmp/results"
)
```

**How to adapt for inference-autotuner**:
- Generate deployment manifests from experiment results
- Output framework-specific launch commands
- Create reproducible deployment packages

**Benefits**:
- Faster deployment of optimal configurations
- Reduced manual configuration errors
- Reproducible deployments

**Code location**: `aiconfigurator/src/aiconfigurator/generator/`

---

## Implementation Priority Matrix

| Feature | Value | Effort | Priority | Status in Autotuner |
|---------|-------|--------|----------|---------------------|
| Pareto Analysis | ⭐⭐⭐⭐⭐ | Medium | **HIGH** | Not implemented |
| Layered Config Factory | ⭐⭐⭐⭐⭐ | Medium | **HIGH** | Partially (task configs) |
| Exponential Penalties | ⭐⭐⭐⭐⭐ | Low | **HIGH** | ✅ Implemented! |
| Backend Strategy | ⭐⭐⭐⭐ | Low | **HIGH** | ✅ Similar pattern exists |
| Performance Modeling | ⭐⭐⭐⭐ | High | **MEDIUM** | Not implemented |
| Config Profiles | ⭐⭐⭐⭐ | Low | **MEDIUM** | ✅ Frontend presets exist |
| Multi-Experiment Compare | ⭐⭐⭐ | Medium | **MEDIUM** | Partial (sequential only) |
| Template Generation | ⭐⭐⭐ | Medium | **LOW** | Not implemented |
| Heterogeneous Deploy | ⭐⭐⭐ | High | **LOW** | Not implemented |
| Replica Scaling | ⭐⭐⭐ | Medium | **LOW** | Not implemented |

---

## Quick Wins (Low Effort, High Value)

### 1. Pareto Analysis Integration
**Effort**: Medium (2-3 days)
**Value**: Very High

**Implementation**:
```python
# In src/orchestrator.py after experiments complete:

def find_pareto_frontier(experiments, objectives):
    """Find non-dominated experiment configurations."""
    pareto_set = []
    for exp in experiments:
        is_dominated = False
        for other in experiments:
            if dominates(other, exp, objectives):
                is_dominated = True
                break
        if not is_dominated:
            pareto_set.append(exp)
    return pareto_set

# Mark pareto-optimal experiments in database
pareto_experiments = find_pareto_frontier(
    experiments,
    objectives={"throughput": "maximize", "latency": "minimize"}
)
```

**Frontend**: Add "Pareto Optimal" badge to experiment results

---

### 2. Layered SLO Configuration
**Effort**: Low (1-2 days)
**Value**: High

**Implementation**:
```python
# In src/utils/config_factory.py (new file):

@dataclass
class SLOLayer:
    name: str
    data: dict
    condition: Optional[Callable] = None

# Define layers
base_slo = SLOLayer("base", {...})
strict_latency = SLOLayer("strict_latency", {"ttft.weight": 3.0})
model_specific = SLOLayer("llama_slo", {...}, condition=lambda ctx: "llama" in ctx.model)

# Merge layers
final_slo = merge_layers([base_slo, strict_latency, model_specific])
```

**Frontend**: Add SLO profile selector in NewTask wizard

---

### 3. Configuration Profiles
**Effort**: Low (1 day)
**Value**: Medium-High

**Implementation**:
- Extend existing frontend presets to include SLO profiles
- Add backend validation for profile schemas
- Document common profiles in examples/

---

## Code Reuse Opportunities

### Direct Code Adaptation
1. **SLO scoring logic**: `aiconfigurator/sdk/pareto_analysis.py` → `src/utils/slo_scoring.py`
2. **Config merging**: `aiconfigurator/sdk/task.py` → `src/utils/config_factory.py`
3. **Pareto frontier**: `aiconfigurator/sdk/pareto_analysis.py` → `src/utils/pareto.py`

### Design Pattern Adoption
1. **Layered config factory** for SLO composition
2. **Operation-based modeling** for performance database
3. **Template generation** for deployment artifacts

---

## Architectural Insights

### aiconfigurator's Key Design Principles

1. **Separation of Concerns**:
   - Configuration layer (task.py)
   - Optimization layer (pareto_analysis.py)
   - Execution layer (backends/)
   - Generation layer (generator/)

2. **Flexible Composition**:
   - Layers stack declaratively
   - Conditional application based on context
   - No hard-coded assumptions

3. **Multi-Level Abstraction**:
   - User-facing: Simple CLI/UI
   - Middle layer: Config factory, profiles
   - Low-level: Operation modeling, backends

**Recommendation**: Inference-autotuner could benefit from similar layering:
```
User Interface (Frontend/CLI)
    ↓
Task Configuration Layer (with profiles, layers)
    ↓
Orchestration Layer (orchestrator.py)
    ↓
Execution Layer (controllers/)
    ↓
Backend Layer (frameworks: sglang, vllm)
```

---

## Detailed Integration Plan

### Phase 1: Core Enhancements (1-2 weeks)
- [ ] Implement Pareto frontier analysis
- [ ] Add layered SLO configuration factory
- [ ] Create SLO profile presets
- [ ] Add Pareto visualization to frontend

### Phase 2: Performance Modeling (2-3 weeks)
- [ ] Build performance database from historical experiments
- [ ] Implement basic operation-based modeling
- [ ] Add performance prediction API
- [ ] Enable "what-if" analysis in frontend

### Phase 3: Advanced Features (3-4 weeks)
- [ ] Multi-experiment comparison dashboard
- [ ] Template-based config generation
- [ ] Heterogeneous deployment support
- [ ] Replica-based scaling model

---

## References

### Key Files to Study
1. **Configuration Factory**: `aiconfigurator/src/aiconfigurator/sdk/task.py`
2. **Pareto Analysis**: `aiconfigurator/src/aiconfigurator/sdk/pareto_analysis.py`
3. **Backend Pattern**: `aiconfigurator/src/aiconfigurator/sdk/backends/base_backend.py`
4. **Performance Modeling**: `aiconfigurator/src/aiconfigurator/sdk/operations.py`
5. **Code Generation**: `aiconfigurator/src/aiconfigurator/generator/api.py`

### Documentation
- Full analysis: `/root/work/inference-autotuner/AICONFIGURATOR_ANALYSIS.md`
- Quick reference: `/root/work/inference-autotuner/AICONFIGURATOR_SUMMARY.md`
- Navigation guide: `/root/work/inference-autotuner/AICONFIGURATOR_INDEX.md`

---

## Summary

**aiconfigurator** provides proven patterns for:
1. Multi-objective optimization (Pareto analysis)
2. Flexible configuration composition (layered factory)
3. Constraint-aware scoring (exponential penalties)
4. Performance prediction (operation-based modeling)

**Immediate action items**:
1. Implement Pareto frontier analysis (highest value)
2. Add layered SLO configuration (improves flexibility)
3. Create SLO profile presets (better UX)

**Long-term opportunities**:
1. Performance modeling database (reduce benchmarking cost)
2. Template-based deployment generation (faster deployment)
3. Heterogeneous hardware support (cost optimization)
