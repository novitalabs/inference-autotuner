# aiconfigurator: Comprehensive Project Analysis

## Executive Summary

**aiconfigurator** is an AI-driven system for automatically configuring disaggregated LLM inference deployments. It searches thousands of parameter configurations using performance modeling and returns optimal deployment setups for high throughput and low latency inference under SLA constraints.

**Key Value Proposition**: 
- Reduces inference deployment configuration from manual trial-and-error to automated optimization
- Achieves 1.7x-2x performance improvements by comparing aggregated vs. disaggregated serving architectures
- Generates production-ready framework configurations for Dynamo deployment

---

## 1. PROJECT OVERVIEW & PURPOSE

### What It Does
aiconfigurator helps deploy LLM inference at scale by automatically finding the best configuration for:
- **Parallelism strategies**: Tensor Parallel (TP), Pipeline Parallel (PP), Data Parallel (DP), Expert Parallel (EP)
- **Quantization modes**: FP16, FP8, FP8-Block, INT8, INT4, NVFP4
- **Serving architectures**: Aggregated (continuous batching) vs. Disaggregated (separate prefill/decode workers)
- **Worker deployment**: Number and GPU allocation per prefill/decode worker
- **Batch sizes**: Context and generation batch sizes
- **Performance targets**: SLA constraints (TTFT, TPOT, latency percentiles)

### Problem Solved
LLM inference deployment is complex:
```
Manual approach: Try configurations -> Benchmark -> Compare -> Repeat
Time: Days to weeks per model/hardware combination

aiconfigurator approach: Specify model + GPU count + SLAs -> Get best config in minutes
```

### Supported Modes
1. **CLI (Default Mode)**: Simple 3-argument configuration comparison (agg vs disagg)
2. **CLI (Exp Mode)**: Complex YAML-driven multi-experiment setup
3. **Web UI (Gradio-based)**: Interactive visualization and parameter exploration
4. **Eval Mode**: Full pipeline (config generation → deployment → benchmarking → analysis)
5. **SDK API**: Programmatic access to all functionality

---

## 2. ARCHITECTURE & CORE COMPONENTS

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
├──────────────────────┬──────────────────┬──────────────────────┤
│ CLI (default/exp)    │   Gradio WebUI   │   SDK (Python API)   │
└──────────────────────┼──────────────────┴──────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│              Task Configuration & Factory Layer                 │
├─────────────────────────────────────────────────────────────────┤
│ • TaskConfigFactory (layered config merging)                   │
│ • Config Profiles (fp8_default, float16_default, etc.)         │
│ • YAML patch/replace modes                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│              Optimization & Analysis Layer                      │
├─────────────────────────────────────────────────────────────────┤
│ • Pareto Analysis (agg_pareto, disagg_pareto)                  │
│ • InferenceSession (aggregated scheduling)                      │
│ • DisaggInferenceSession (prefill/decode separation)            │
│ • Parallel Config Enumeration                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│              Performance Modeling Layer                          │
├─────────────────────────────────────────────────────────────────┤
│ • Model Classes (GPTModel, LLAMAModel, MOEModel, etc.)          │
│ • Operation Classes (GEMM, Attention, AllReduce, NCCL, etc.)    │
│ • Backend Adapters (TRTLLM, SGLang, VLLM)                      │
│ • PerfDatabase (CSV/interpolation-based performance data)       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│              Code Generation Layer                              │
├─────────────────────────────────────────────────────────────────┤
│ • Backend Generators (TRTLLM, SGLang, VLLM)                    │
│ • Artifact Writers (Kubernetes YAML, shell scripts, configs)    │
└──────────────────────────────────────────────────────────────────┘
```

### Key Modules

#### 1. **Core SDK** (`src/aiconfigurator/sdk/`)
- **`common.py`**: Enums and data structures (model families, quantization modes, etc.)
- **`models.py`**: Model abstractions (GPTModel, LLAMAModel, MOEModel, DeepSeekModel, etc.)
- **`config.py`**: Configuration dataclasses (ModelConfig, RuntimeConfig)
- **`operations.py`**: Operation abstractions (GEMM, Attention, AllReduce, P2P, NCCL)
- **`perf_database.py`**: Performance data loading and querying
- **`inference_session.py`**: Aggregated and disaggregated inference simulation
- **`pareto_analysis.py`**: Pareto frontier computation and configuration search
- **`task.py`**: Task configuration factory with layered merging
- **`backends/`**: Backend adapters (TRTLLM, SGLang, VLLM)

#### 2. **CLI Interface** (`src/aiconfigurator/cli/`)
- **`main.py`**: Argument parsing and orchestration
- **`report_and_save.py`**: Results formatting and file generation
- `example.yaml`: Template configurations
- `exps/`: Pre-built experiment templates (heterogeneous, quantization variants, etc.)

#### 3. **Web UI** (`src/aiconfigurator/webapp/`)
- **`main.py`**: Gradio interface setup
- **`components/`**: Tab components (static, agg, disagg, pareto comparison)
- **`events/`**: Event handlers and data flow

#### 4. **Code Generator** (`src/aiconfigurator/generator/`)
- **`api.py`**: Public API for artifact generation
- **`backends/`**: Framework-specific generators (TRTLLM, SGLang, VLLM)
- **`inputs/`**: Configuration parsing and validation
- **`templates/`**: Jinja2 templates for Kubernetes YAML, shell scripts
- **`utils/`**: Node allocation, artifact writers

#### 5. **Data Collection** (`collector/`)
- **`collect.py`**: GEMM, attention, MoE performance profiling
- **`collect_nccl.py`**: Communication operation profiling
- **`collect_all_reduce.py`**: Custom AllReduce profiling
- **Framework-specific collectors**: TRTLLM, SGLang, VLLM

---

## 3. KEY FEATURES & CAPABILITIES

### 3.1 Supported Models
- **Dense Models**: LLaMA 2/3.1 (7B-405B), GPT, Qwen (0.6B-480B)
- **Mixture-of-Experts (MoE)**: Mixtral 8x7B/8x22B, Qwen3 MoE variants, DeepSeek-V3
- **Architecture Variants**: Standard attention (MHA/GQA), Multi-Head Latent Attention (MLA), MoE with fine-grained expert selection
- **Extension**: Custom model registration framework

### 3.2 Configuration Search Space

**Parallelism Dimensions**:
```python
tp_list = [1, 2, 4, 8]              # Tensor parallel
pp_list = [1, 2, 4]                 # Pipeline parallel
dp_list = [1, 2, 4, 8]              # Data parallel (attention)
moe_tp_list = [1, 2, 4]             # MoE tensor parallel
moe_ep_list = [1, 2, 4, 8, 16]      # Expert parallel
```

**Quantization Combinations**:
- GEMM: FP16, FP8, FP8-Block, INT8 WO, INT4 WO, NVFP4
- KV Cache: FP16, FP8, INT8
- Attention: FP16, FP8
- Communication: Half-precision
- MoE: FP16, FP8, FP8-Block, W4A-FP8, INT4 WO, NVFP4

**Serving Architecture**:
- **Aggregated (Agg)**: Single worker pool with continuous batching
- **Disaggregated (Disagg)**: Separate prefill and decode worker pools with static batching
- **Heterogeneous**: Different systems for prefill vs. decode (H200 prefill + H100 decode)

### 3.3 Constraint Handling (SLA-Aware Optimization)

**Metrics Tracked**:
- TTFT (Time to First Token): Latency of first token generation
- TPOT (Time Per Output Token): Generation speed
- P50/P90/P99 latency percentiles
- Throughput (tokens/s/GPU)

**Constraint Enforcement**:
- Hard failures: Experiment marked FAILED if constraint exceeded
- Soft penalties: Score degradation for SLA violations
- Fail ratio: Allow N% of requests to exceed threshold

**Example Configuration**:
```yaml
slo:
  ttft:
    threshold: 300.0    # milliseconds
    weight: 2.0
    hard_fail: false
  tpot:
    threshold: 10.0
    weight: 2.0
    hard_fail: false
  latency:
    p90:
      threshold: 5.0
      weight: 2.0
      hard_fail: true
      fail_ratio: 0.2    # Fail if >20% exceed threshold
```

### 3.4 Performance Modeling

**Approach**: Compose operation-level performance estimates into end-to-end predictions

**Operations Modeled**:
```
Context Phase (Prefill):
├── GEMM (Linear layers)
├── Attention (Query-Key-Value projections, attention computation)
├── KV Cache (Memory operations)
├── AllReduce (Tensor parallel reduction)
├── Embedding (Token → embedding)
└── MoE (Expert routing and computation)

Generation Phase (Decode):
├── GEMM (single-token inference)
├── Attention (KV cache retrieval + attention)
├── Sampling/Beam search
├── MoE (with expert routing)
└── P2P (Pipeline parallel communication)
```

**Data Sources**:
- Pre-collected **CSV databases** with measured operation timings
- **Interpolation** for unseen batch sizes/sequence lengths
- **Extrapolation** using scipy.interpolate functions

**Latency Correction**:
```python
latency_correction_scale: float  # Empirical adjustment factor
actual_latency = estimated_latency * latency_correction_scale
```

### 3.5 Specialized Features

**Multi-Token Prediction (MTP) / Speculative Decoding**:
```yaml
nextn: 2  # 2 draft tokens per iteration
nextn_accept_rates: [0.85, 0.3, 0, 0, 0]  # Acceptance rate per draft token position
```

**Wide Expert Parallel** (for MoE models):
```python
enable_wide_ep: true  # Enable fine-grained expert parallelization
```

**Advanced Tuning Config**:
```yaml
advanced_tuning_config:
  prefill_latency_correction_scale: 1.1
  decode_latency_correction_scale: 1.08
  prefill_max_batch_size: 1
  decode_max_batch_size: 512
```

---

## 4. DESIGN PATTERNS & ARCHITECTURE INSIGHTS

### 4.1 Layered Configuration Factory Pattern

**Problem**: Configuration composition from multiple sources (defaults, profiles, YAML patches)

**Solution**: `TaskConfigFactory` with explicit layer composition

```python
@dataclass(frozen=True)
class ConfigLayer:
    name: str
    data: dict | Callable[[TaskContext], dict]
    condition: Callable[[TaskContext], bool] | None = None
    
    def applies_to(self, ctx: TaskContext) -> bool:
        if self.condition is None:
            return True
        return self.condition(ctx)
    
    def resolve(self, ctx: TaskContext) -> dict:
        payload = self.data(ctx) if callable(self.data) else self.data
        return copy.deepcopy(payload)
```

**Benefits**:
- Declarative configuration inheritance
- Conditional layer application based on context (serving mode, model type)
- Clear separation of concerns
- Easy to extend with new profiles

**Usage Pattern**:
```python
# Layer 1: Base configuration (loaded from YAML)
# Layer 2: Mode-specific adjustments (agg vs disagg)
# Layer 3: Backend-specific tuning (TRTLLM vs SGLang)
# Layer 4: User profiles (fp8_default, float16_default)
# Layer 5: YAML patch/replace from CLI
```

### 4.2 Strategy Pattern for Backends

**Backend Abstraction**:
```python
class BaseBackend(ABC):
    @abstractmethod
    def run_static(...) -> InferenceSummary: ...
    
    @abstractmethod
    def run_agg(...) -> InferenceSummary: ...
    
    @abstractmethod
    def find_best_agg_result_under_constraints(...): ...
```

**Implementations**:
- **TRTLLM**: TensorRT-LLM specific performance modeling
- **SGLang**: SGLang runtime behavior
- **VLLM**: VLLM inference simulation

**Factory Pattern**:
```python
backend = get_backend(backend_name)  # Returns correct implementation
sess = InferenceSession(model, database, backend)
```

### 4.3 Model Hierarchy

**Base Model Class**:
```python
class BaseModel:
    def __init__(self, ...):
        # Model properties
        self.layers = layers
        self.d = d  # Head dimension
        self.hidden = hidden
        self.inter = inter
        
        # Build operation lists (context, generation, etc.)
        self.context_ops = [GEMM, Attention, Embedding, ...]
        self.generation_ops = [...]
```

**Specializations**:
- `GPTModel`: Standard transformer architecture
- `LLAMAModel`: LLaMA architecture with GQA
- `MOEModel`: Mixture of Experts with expert routing
- `DeepSeekModel`: MLA (Multi-Head Latent Attention)

**Benefits**:
- Clear model property specification
- Architecture-specific operation sequences
- Quantization and parallelism-aware computation

### 4.4 Database & Interpolation Strategy

**Performance Database**:
```python
class PerfDatabase:
    def query_gemm(quant_mode: GEMMQuantMode, tp_size: int, m: int, n: int, k: int) -> float
    def query_attention(quant_mode: FMHAQuantMode, tp_size: int, batch_size: int) -> float
    def query_nccl(comm_quant_mode, num_gpus, nccl_op, message_size) -> float
```

**Data Storage**:
- **CSV files** with operation timings on specific systems/frameworks
- **Hierarchical organization**: `systems/{system_name}/{backend}/{version}/{operation}.txt`

**Query Resolution**:
```
1. Direct lookup in CSV
2. If not found: Interpolate from nearby data points (scipy.interpolate)
3. If extrapolation needed: Use extrapolation methods
4. Fallback: Log warning and use cached estimate
```

### 4.5 Pareto Analysis & Multi-Objective Optimization

**Search Objectives**:
```python
# Objective function combines:
score = (throughput_weight × throughput) - (latency_weight × latency)

# Constraint filtering:
if ttft > threshold:
    score *= exponential_penalty(violation_ratio, steepness=0.1)
```

**Pareto Frontier Generation**:
```python
def agg_pareto(model, runtime_config, database, backend, parallel_config_list):
    # For each parallel config (TP, PP, DP, MoE_TP, MoE_EP)
    #   For each batch size
    #     Run inference simulation
    #     Check SLA constraints
    #     If valid: add to results
    # Return dominated configurations removed
```

**Result Set**:
```
Pareto Frontier = {configs where no other config is strictly better in all objectives}
```

### 4.6 Inference Simulation Modes

**Aggregated Serving**:
```
Input: Continuous stream of requests
Model: Batch tokens across multiple requests in-flight
Latency: Depends on batch size (amortizes overhead)
Use Case: High throughput, relaxed latency requirements
```

**Disaggregated Serving**:
```
Prefill Phase: Process request's input tokens (high compute, high latency variance)
Decode Phase: Generate output tokens one-by-one (low compute, predictable latency)
Architecture: Separate prefill worker pool + decode worker pool
Benefit: Decouples prefill latency from decode throughput
Use Case: Tight SLA requirements (low TTFT, consistent TPOT)
```

**Static Batching** (used in disagg simulation):
```python
def run_static(model, database, runtime_config, mode, stride):
    context_latency = simulate_prefill_phase(runtime_config.isl)
    generation_latency = simulate_decode_phase(runtime_config.osl, stride)
    return context_latency + generation_latency
```

---

## 5. CONFIGURATION MANAGEMENT APPROACH

### 5.1 YAML Configuration Structure

**Top-level Sections**:
```yaml
exps:                    # List of experiments to run
  - exp_disagg_full
  - exp_agg_simplified

exp_disagg_full:         # Experiment definition
  mode: "patch"          # patch or replace
  serving_mode: "disagg" # disagg or agg
  model_name: "QWEN3_32B"
  total_gpus: 32
  system_name: "h200_sxm"
  decode_system_name: "h200_sxm"  # Optional
  backend_name: "trtllm"
  backend_version: "1.0.0rc3"
  isl: 4000              # Input sequence length
  osl: 1000              # Output sequence length
  ttft: 1000.0           # TTFT SLA in ms
  tpot: 40.0             # TPOT SLA in ms
  enable_wide_ep: false
  profiles: ["fp8_default"]
  config:                # Override defaults
    nextn: 1
    nextn_accept_rates: [0.85, 0, 0, 0, 0]
    prefill_worker_config:
      gemm_quant_mode: "fp8_block"
      moe_quant_mode: "fp8_block"
      kvcache_quant_mode: "float16"
      fmha_quant_mode: "float16"
      comm_quant_mode: "half"
      num_gpu_per_worker: [4, 8]
      tp_list: [1, 2, 4, 8]
      pp_list: [1]
      dp_list: [1]
      moe_tp_list: [1]
      moe_ep_list: [1, 2, 4, 8]
    decode_worker_config: {...}
    replica_config:
      num_gpu_per_replica: [8, 16, 24, 32, ...]
      max_gpu_per_replica: 128
      max_prefill_worker: 32
      max_decode_worker: 32
    advanced_tuning_config:
      prefill_latency_correction_scale: 1.1
      decode_latency_correction_scale: 1.08
```

### 5.2 Configuration Modes

**Default Mode** (CLI):
```bash
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm
# Automatically creates agg and disagg experiments, compares them
```

**Exp Mode** (YAML-driven):
```bash
aiconfigurator cli exp --yaml_path config.yaml
# Runs custom experiments defined in YAML
```

**Patch vs Replace**:
- `mode: patch`: YAML config merges with defaults (only specified fields override)
- `mode: replace`: YAML config completely replaces defaults

### 5.3 Profile System

**Pre-built Profiles**:
- `fp8_default`: FP8 quantization for all operations
- `float16_default`: FP16 (no quantization)
- `nvfp4_default`: NVIDIA FP4 quantization

**Custom Profiles**:
```python
@dataclass(frozen=True)
class ConfigLayer:
    name: "custom_profile"
    data: {"gemm_quant_mode": "nvfp4", ...}
    condition: lambda ctx: ctx.model_name == "DEEPSEEK_V3"
```

---

## 6. API DESIGN & INTERFACES

### 6.1 CLI Interface

```bash
# Main entry points
aiconfigurator cli default --model MODEL --total_gpus N --system SYSTEM
aiconfigurator cli exp --yaml_path CONFIG.yaml
aiconfigurator webapp
aiconfigurator eval --config CONFIG.yaml --save_dir RESULTS

# Common options
--ttft 300              # TTFT SLA in ms
--tpot 10               # TPOT SLA in ms
--isl 4000              # Input sequence length
--osl 500               # Output sequence length
--save_dir DIR          # Save results and generated configs
--debug                 # Enable debug logging
--backend trtllm        # Inference framework
--backend_version 1.0.0rc3  # Framework version
```

### 6.2 SDK/Programmatic API

```python
from aiconfigurator.sdk import task

# Create task context
ctx = task.TaskContext(
    serving_mode="disagg",
    model_name="QWEN3_32B",
    system_name="h200_sxm",
    decode_system_name="h200_sxm",
    backend_name="trtllm",
    backend_version="1.0.0rc3",
    isl=4000,
    osl=1000,
    ttft=300.0,
    tpot=10.0,
    total_gpus=32,
    profiles=["fp8_default"],
    yaml_patch={...},
)

# Generate configuration
config, applied_layers = task.TaskConfigFactory.create(ctx)

# Run inference simulation
from aiconfigurator.sdk.pareto_analysis import disagg_pareto
results = disagg_pareto(
    model_name="QWEN3_32B",
    runtime_config=runtime_config,
    database=perf_db,
    backend_name="trtllm",
    model_config=model_config,
    parallel_config_list=[...],
)

# Generate framework configs
from aiconfigurator.generator.api import generate_backend_config
artifacts = generate_backend_config.from_runtime(
    cfg=config,
    backend="trtllm",
    version="1.0.0rc6",
    save_dir="/tmp/results"
)
```

### 6.3 Web UI API (Gradio-based)

**Features**:
- Interactive parameter adjustment
- Real-time Pareto frontier visualization
- Configuration saving and comparison
- Results export

**Components**:
```
Static Tab (Basic configuration)
├── Model selector
├── GPU count input
├── System selector
├── SLA parameters (TTFT, TPOT)
└── Run button

Agg Tab (Aggregated serving config)
├── Parallel config selector
├── Quantization options
├── Batch size tuning
└── Results table

Disagg Tab (Disaggregated serving)
├── Prefill/decode worker config
├── Replica configuration
├── Scheduler options
└── Pareto visualization

Pareto Comparison Tab
└── Multi-run comparison plots
```

---

## 7. CODE GENERATION & DEPLOYMENT

### 7.1 Artifact Generation Pipeline

```
Configuration (YAML/Dict)
    ↓
InputParser (Parse & validate)
    ↓
GeneratorContext (Internal representation)
    ↓
Backend-specific Generator (TRTLLM, SGLang, VLLM)
    ↓
ArtifactBundle (Generated artifacts)
    ↓
Writers (Save to disk)
    ↓
Output Structure:
    results/
    ├── agg/
    │   ├── best_config_topn.csv
    │   ├── config.yaml
    │   ├── pareto.csv
    │   └── top1/
    │       ├── agg/
    │       │   ├── agg_config.yaml          # Framework config
    │       │   ├── k8s_deploy.yaml          # Kubernetes manifest
    │       │   └── node_0_run.sh            # Deployment script
    │       └── generator_config.yaml        # Generation metadata
    └── disagg/
        ├── best_config_topn.csv
        ├── config.yaml
        ├── pareto.csv
        └── top1/
            ├── disagg/
            │   ├── prefill_config.yaml
            │   ├── decode_config.yaml
            │   ├── k8s_deploy.yaml
            │   └── node_0_run.sh
            └── generator_config.yaml
```

### 7.2 Backend Generators

**TRTLLM Generator**:
- Generates `agg_config.yaml` / `prefill_config.yaml` + `decode_config.yaml`
- Creates shell scripts for model serving
- Outputs Kubernetes deployment manifests

**SGLang Generator**:
- SGLang-specific configuration format
- Handles MLA (Multi-Head Latent Attention) specific settings

**VLLM Generator**:
- VLLM-compatible configuration
- Handles OpenAI API exposure

### 7.3 Template System

**Jinja2 Templates** for artifact generation:
```jinja2
# templates/trtllm/agg_config.yaml.j2
model_dir: "{{ model_path }}"
tensor_parallel_size: {{ tp_size }}
pipeline_parallel_size: {{ pp_size }}
dtype: "{{ dtype }}"
kv_cache_dtype: "{{ kv_cache_dtype }}"
max_batch_size: {{ batch_size }}
```

**Template Variables**:
- Model properties (layers, hidden size, vocab size)
- Parallelism configuration (TP, PP, DP, MoE)
- Quantization choices
- System configuration

---

## 8. TECHNOLOGY STACK

### Core Technologies
- **Language**: Python 3.9+
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Plotext, Bokeh, Recharts
- **Web Framework**: Gradio 5.47.1 (UI), FastAPI (future REST API)
- **Configuration**: YAML, Pydantic, Munch
- **Templating**: Jinja2
- **CLI**: argparse (no heavy framework)
- **Build**: setuptools, pyproject.toml

### Key Dependencies
```toml
fastapi>=0.115.12          # Future REST API support
gradio==5.47.1             # Web UI
pydantic~=2.11.4           # Config validation
pyyaml>=6.0                # Configuration files
scipy>=1.13.1              # Interpolation
pandas>=2.2.3              # Data analysis
numpy~=1.26.4              # Numerical computing
plotext>=5.3.2             # Terminal plotting
matplotlib>=3.9.4          # Static plots
```

### Development Tools
- **Linting**: Ruff (high-performance linter)
- **Testing**: Pytest with plugins (asyncio, cov, xdist)
- **CI/CD**: GitHub Actions
- **Package Management**: uv (fast Python package installer)

---

## 9. FEATURES THAT COULD BENEFIT INFERENCE AUTOTUNER

### 9.1 Layered Configuration Factory
**How it applies**:
- Inference autotuner could use similar pattern for SLO configuration composition
- Layer structure: Base SLO → Model-specific SLO → Experiment-specific SLO

**Implementation**:
```python
class SLOConfigLayer:
    name: str
    data: dict | Callable[[Context], dict]
    condition: Optional[Callable] = None
    
# Apply in sequence to build final SLO config
```

### 9.2 Pareto Analysis Framework
**How it applies**:
- Multi-objective optimization (throughput vs latency vs cost)
- Constraint satisfaction with soft/hard penalties
- Visualization of trade-offs

**Integration**:
```python
# Find configurations satisfying user-specified Pareto preferences
best_configs = find_pareto_frontier(
    results_df,
    objectives={"throughput": "maximize", "latency": "minimize"},
    constraints={"cost": "< 1000"},
)
```

### 9.3 Performance Modeling Abstractions
**How it applies**:
- Operation-based performance prediction
- Backend abstraction for different runtimes
- Interpolation for unseen configurations

**Adaptation**:
```python
class InferenceOperation(ABC):
    def estimate_latency(self, config: Config) -> float: ...
    def estimate_memory(self, config: Config) -> float: ...
    
class TensorParallelGEMM(InferenceOperation): ...
class AllReduceOverhead(InferenceOperation): ...
```

### 9.4 SLA Constraint Handling
**Current implementation in aiconfigurator**:
```yaml
slo:
  ttft:
    threshold: 300.0
    weight: 2.0
    hard_fail: false
  latency:
    p90:
      threshold: 5.0
      weight: 2.0
      hard_fail: true
      fail_ratio: 0.2
```

**Integration pattern**:
- Reuse SLO scoring mechanism
- Extend to support inference-specific metrics
- Apply exponential penalty functions for violations

### 9.5 Backend Strategy Pattern
**How it applies**:
- Abstract controller implementations (Docker vs Kubernetes)
- Unified interface for benchmarking backends
- Easy addition of new inference engines

**Current inference-autotuner pattern**:
```python
class BaseModelController(ABC):
    @abstractmethod
    def deploy(self, ...): ...
    @abstractmethod
    def benchmark(self, ...): ...

class DockerController(BaseModelController): ...
class OMEController(BaseModelController): ...
```

**Enhancement opportunity**:
- Use aiconfigurator's model abstraction for performance estimation
- Combine real benchmarking with predicted performance

### 9.6 Configuration Hierarchies
**How it applies**:
- Task-level → Experiment-level → Run-level configuration
- YAML patch/replace modes for flexible overrides
- Profile-based configuration presets

**Inference autotuner use case**:
```yaml
# Task-level defaults
base_config:
  model: llama-3-2-1b
  
# Experiment-level overrides
experiments:
  - name: "exp1"
    config_patch:
      tp-size: [1, 2]
      mem-fraction-static: [0.7]
```

### 9.7 Result Aggregation & Comparison
**How it applies**:
- Multi-experiment result analysis
- CSV export for statistical analysis
- Visualization of Pareto frontiers

**Inference autotuner integration**:
```python
# Compare multiple inference engines on same hardware
results_df = compare_frameworks(
    frameworks=["sglang", "vllm"],
    model="qwen3-32b",
    parameters={...},
)
pareto_frontier = extract_pareto(results_df)
```

### 9.8 Performance Data Management
**How it applies**:
- Centralized performance database concept
- Interpolation for unseen configurations
- Version management for framework updates

**Application**:
```python
# Offline database + online refinement
offline_db = load_database("frameworks/sglang/v0.5.2")
refined_db = update_with_benchmarks(offline_db, recent_measurements)
estimated_latency = refined_db.query(config, metric="ttft")
```

### 9.9 Automation & End-to-End Pipelines
**How it applies**:
- Scripted experiment execution
- Automatic artifact generation
- Result visualization and reporting

**Inference autotuner enhancement**:
- Extend `tools/automation/` concept for inference tuning
- Combine configuration generation → deployment → benchmarking → analysis

### 9.10 Code Generation for Different Runtimes
**How it applies**:
- Template-based configuration generation
- Multi-format support (Kubernetes YAML, shell scripts, Docker)
- Version-aware template selection

**Inference autotuner benefit**:
- Generate framework-specific benchmarking commands
- Output deployment manifests automatically
- Support new inference engines by adding generators

---

## 10. UNIQUE & INNOVATIVE FEATURES

### 10.1 Disaggregated Inference Modeling
**Innovation**: Separate modeling of prefill and decode phases
- **Problem**: Traditional serving bundles prefill + decode latency
- **Solution**: Model as independent worker pools with different optimization objectives
- **Result**: Can achieve 1.7x better throughput under SLA

### 10.2 Multi-Hardware Support
**Innovation**: Heterogeneous deployments (different GPUs for prefill vs decode)
```yaml
system_name: "h200_sxm"         # Prefill on H200 (compute-optimized)
decode_system_name: "h100_sxm"  # Decode on H100 (cheaper)
```

### 10.3 Fine-Grained Quantization Control
**Innovation**: Per-component quantization with operation-level impact modeling
```
GEMM: fp8_block (highest impact)
Attention: float16 (sensitive to precision)
KV Cache: fp8 (large memory impact)
Communication: half (network-optimized)
```

### 10.4 Model Family Abstraction
**Innovation**: Pluggable model architectures with shared operation interface
- Support for emerging model types (MLA, MoE variants)
- Consistent quantization + parallelism semantics across families
- Easy model addition through registering operations

### 10.5 Replica-Based Scaling
**Innovation**: Treats disaggregated deployments as xPyD replicas
```
Replica = x prefill workers (tp=1, pp=1) + y decode workers (tp=4, pp=1)
Scale to total_gpus by running N replicas
```
- Simplifies complex heterogeneous deployments
- Enables predictable scaling characteristics

### 10.6 Constraint-Aware Scoring
**Innovation**: Exponential penalty function for SLA violations
```python
if violation_ratio > 0:
    penalty = weight × exp(violation_ratio / steepness)
    score *= (1 - penalty)
```
- Soft constraints (multiply score) vs hard constraints (fail experiment)
- Tunable steepness for constraint sensitivity

---

## 11. PROJECT STRUCTURE (Detailed)

```
aiconfigurator/
├── README.md                                # Main documentation
├── DEVELOPMENT.md                           # Developer guide
├── CONTRIBUTING.md                          # Contribution guidelines
├── CODE_OF_CONDUCT.md                       # Community standards
├── LICENSE (Apache-2.0)                     # Licensing
├── pyproject.toml                           # Package configuration
├── pytest.ini                               # Test configuration
├── docker/                                  # Docker build artifacts
│   └── Dockerfile                           
├── src/aiconfigurator/
│   ├── __init__.py
│   ├── main.py                              # Entry point dispatcher
│   ├── cli/
│   │   ├── main.py                          # CLI orchestration
│   │   ├── report_and_save.py              # Results formatting
│   │   ├── example.yaml                     # Template config
│   │   └── exps/                            # Pre-built experiments
│   │       ├── hetero_disagg.yaml           # H200 vs B200
│   │       └── qwen3_32b_disagg_pertensor.yaml
│   ├── sdk/
│   │   ├── common.py                        # Enums & constants
│   │   ├── config.py                        # Config dataclasses
│   │   ├── models.py                        # Model architectures
│   │   ├── operations.py                    # Operation abstractions
│   │   ├── task.py                          # Config factory
│   │   ├── inference_session.py             # Inference simulation
│   │   ├── pareto_analysis.py               # Optimization engine
│   │   ├── perf_database.py                 # Performance data
│   │   ├── inference_summary.py             # Results structure
│   │   ├── utils.py                         # Utilities
│   │   └── backends/                        # Backend adapters
│   │       ├── base_backend.py
│   │       ├── factory.py
│   │       ├── trtllm_backend.py
│   │       ├── sglang_backend.py
│   │       └── vllm_backend.py
│   ├── generator/
│   │   ├── api.py                           # Generator entry point
│   │   ├── types.py                         # Type definitions
│   │   ├── cli_args.py                      # CLI argument helpers
│   │   ├── cfg_example.yaml
│   │   ├── inputs/
│   │   │   ├── schema.py                    # Config schemas
│   │   │   └── parser.py                    # Config parsing
│   │   ├── backends/
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   ├── trtllm.py
│   │   │   ├── sglang.py
│   │   │   └── vllm.py
│   │   ├── templates/                       # Jinja2 templates
│   │   │   ├── trtllm/
│   │   │   ├── sglang/
│   │   │   └── vllm/
│   │   └── utils/
│   │       ├── node_allocation.py
│   │       └── writers.py
│   ├── webapp/
│   │   ├── main.py                          # Gradio UI setup
│   │   ├── components/
│   │   │   ├── base.py
│   │   │   ├── static_tab.py
│   │   │   ├── agg_tab.py
│   │   │   ├── disagg_pareto_tab.py
│   │   │   ├── agg_pareto_tab.py
│   │   │   ├── pareto_comparison_tab.py
│   │   │   └── readme_tab.py
│   │   ├── events/
│   │   │   ├── event_handler.py
│   │   │   └── event_fn.py
│   │   └── README.md
│   ├── eval/
│   │   ├── main.py                          # End-to-end pipeline
│   │   ├── service.py                       # Service deployment
│   │   ├── pipeline.py                      # Workflow orchestration
│   │   ├── gpu.py                           # GPU monitoring
│   │   ├── utils.py
│   │   └── benchmarks/
│   │       └── genai_perf_runner.py         # Benchmarking
│   └── systems/                             # Performance databases
│       ├── h100_sxm.yaml                    # System definitions
│       ├── h200_sxm.yaml
│       ├── b200_sxm.yaml
│       └── h100_sxm/                        # Database files
│           ├── trtllm/
│           │   ├── 0.20.0/
│           │   │   ├── gemm.txt
│           │   │   ├── attention.txt
│           │   │   ├── comm_perf.txt
│           │   │   └── ...
│           │   └── 1.0.0rc3/
│           └── sglang/
├── collector/                               # Data collection
│   ├── README.md
│   ├── collect.py                           # Main collection script
│   ├── collect_nccl.py
│   ├── collect_all_reduce.py
│   ├── helper.py
│   ├── trtllm/                              # Framework-specific
│   ├── sglang/
│   └── vllm_v1/
├── tools/
│   ├── automation/                          # End-to-end automation
│   │   ├── README.md
│   │   ├── launch_eval.sh
│   │   └── config.env
│   ├── simple_sdk_demo/                     # SDK usage examples
│   │   └── README.md
│   └── sanity_check/                        # Database validation
│       └── README.md
├── docs/
│   ├── cli_user_guide.md                    # CLI documentation
│   ├── advanced_tuning.md                   # Tuning guide
│   ├── dynamo_deployment_guide.md           # Deployment docs
│   └── add_a_new_model.md                   # Model addition guide
├── tests/
│   ├── cli/                                 # CLI tests
│   │   └── e2e_validation/
│   └── sdk/                                 # SDK tests
├── config/                                  # Kubernetes resources (if any)
└── .github/
    ├── workflows/                           # CI/CD
    └── pull_request_template.md
```

---

## 12. DEPENDENCIES & ECOSYSTEM

### Runtime Dependencies
- **Framework Support**: TRTLLM, SGLang, VLLM
- **Hardware**: NVIDIA GPUs (H100, H200, B200, GB200, A100)
- **Deployment**: Kubernetes (optional), Docker, Dynamo (inference engine)
- **Benchmarking**: genai-bench for performance validation

### Related Projects
- **Dynamo**: NVIDIA's inference engine (generates configs for this)
- **genai-bench**: Inference benchmarking tool
- **NCCL**: Communication library
- **TRTLLM**: TensorRT-LLM inference framework

---

## 13. KNOWN LIMITATIONS & OPEN AREAS

### Known Issues (from README)
1. MoE memory estimation needs workspace consideration
2. Results can be overly optimistic in low-speed, high-throughput regions
3. Results are estimates - should be validated with real benchmarks

### Open Opportunities
- WebSocket support for real-time UI updates (currently polling-based)
- Bayesian optimization for search (currently exhaustive grid search)
- Support for additional model architectures
- Dynamic compilation optimization
- Cross-hardware prediction (train on H100, predict for H200)

---

## 14. SUMMARY FOR INFERENCE AUTOTUNER ADAPTATION

### Most Valuable Concepts
1. **Layered Configuration Factory** - Enables flexible SLO composition
2. **Pareto Analysis Framework** - Multi-objective constraint satisfaction
3. **Backend Strategy Pattern** - Clean separation of inference engines
4. **Performance Modeling Abstractions** - Reusable operation-based estimation
5. **Constraint Handling with Penalties** - SLA violation management
6. **Configuration Hierarchies** - Flexible YAML-based customization

### Integration Points
- Use aiconfigurator's SLO model for autotuner's constraint checking
- Reuse Pareto analysis for finding optimal parameter combinations
- Leverage config factory for SLO layer composition
- Adopt backend pattern for supporting new inference frameworks
- Use performance database concept for offline estimation

### Code Reuse Potential
- SLO scoring functions (`aiconfigurator/sdk/pareto_analysis.py`)
- Configuration merging logic (`aiconfigurator/sdk/task.py`)
- Backend abstraction (`aiconfigurator/sdk/backends/base_backend.py`)
- Results aggregation and visualization code

