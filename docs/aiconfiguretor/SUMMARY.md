# aiconfigurator - Quick Reference Summary

**File Location**: `/root/work/inference-autotuner/third_party/aiconfigurator/`

## What Is It?
An NVIDIA AI system for automatically optimizing LLM inference deployments. It searches thousands of configuration combinations (parallelism, quantization, serving architecture) to find the best deployment under SLA constraints.

## Key Innovation: Disaggregated Inference
- Separates prefill (input processing) and decode (output generation) into different worker pools
- Achieves **1.7x-2x better throughput** than traditional aggregated serving under SLA constraints
- Enables tighter control over TTFT (Time to First Token) and TPOT (Time Per Output Token)

## Core Value Proposition
```
Input: Model name + GPU count + GPU type + SLA targets
        ↓
Process: Search thousands of configurations automatically
        ↓
Output: Ready-to-deploy framework configs + performance estimates
        ↓
Benefit: Minutes instead of days to find optimal configuration
```

## Main Components

### 1. **SDK** (`src/aiconfigurator/sdk/`)
Core optimization engine with:
- Model abstractions (LLAMA, GPT, MoE, DeepSeek)
- Operation-level performance modeling (GEMM, Attention, Communication)
- Pareto frontier analysis (multi-objective optimization)
- Inference session simulation (aggregated & disaggregated)

### 2. **CLI** (`src/aiconfigurator/cli/`)
User-friendly command-line interface:
- `default` mode: 3 args → instant agg vs disagg comparison
- `exp` mode: YAML-driven custom experiments

### 3. **Web UI** (`src/aiconfigurator/webapp/`)
Interactive Gradio-based interface for parameter exploration and visualization

### 4. **Code Generator** (`src/aiconfigurator/generator/`)
Generates production-ready framework configurations:
- TRTLLM config files
- Kubernetes deployment manifests
- Shell scripts for deployment

### 5. **Data Collector** (`collector/`)
Offline profiling of operation timings on target hardware

## Usage Examples

### CLI - Default Mode (Simplest)
```bash
aiconfigurator cli default \
  --model QWEN3_32B \
  --total_gpus 32 \
  --system h200_sxm \
  --ttft 300 \
  --tpot 10
```
Output: Comparison showing disagg is 1.7x better

### CLI - Exp Mode (Advanced)
```bash
aiconfigurator cli exp --yaml_path custom_config.yaml
```
Runs custom experiments from YAML (multi-hardware, different quantization, etc.)

### Web UI
```bash
aiconfigurator webapp
# Opens interactive UI at http://localhost:7860
```

### Full Pipeline (Generate + Deploy + Benchmark)
```bash
aiconfigurator eval --config config.yaml --save_dir results/
```

## Configuration Parameters

### SLA Constraints
```yaml
slo:
  ttft:                    # Time to first token
    threshold: 300.0       # milliseconds
    weight: 2.0
    hard_fail: false
  tpot:                    # Time per output token
    threshold: 10.0
    weight: 2.0
    hard_fail: false
  latency:
    p90:
      threshold: 5.0
      weight: 2.0
      hard_fail: true      # Mark experiment FAILED if violated
      fail_ratio: 0.2      # Allow up to 20% requests to exceed
```

### Search Space
- **Parallelism**: TP [1,2,4,8] × PP [1,2,4] × DP [1,2,4,8] × EP [1,2,4,8,16]
- **Quantization**: FP16, FP8, FP8-Block, INT8, INT4, NVFP4 (per-component)
- **Serving Mode**: Aggregated (continuous batching) or Disaggregated (prefill/decode split)
- **Batch Sizes**: Context batch size, Generation batch size
- **Architectures**: Dense, MoE, MLA (Multi-Head Latent Attention)

## Supported Models

### Dense Models
- LLaMA 2/3.1 (7B-405B)
- Qwen 2.5/3 (0.6B-480B)
- GPT variants

### Mixture-of-Experts
- Mixtral 8x7B/8x22B
- Qwen3 MoE variants (30B/235B/480B)
- **DeepSeek-V3** (fine-grained expert selection)

### Specialized Architectures
- Multi-Head Latent Attention (MLA) support
- Fine-grained Expert Parallel (EP) for MoE
- Multi-Token Prediction (speculative decoding)

## Supported Hardware & Frameworks

### GPUs
- H100 SXM, H200 SXM (Hopper)
- B200 SXM, GB200 SXM (Blackwell, in development)
- A100 SXM

### Inference Frameworks
- **TRTLLM** (TensorRT-LLM): Fully supported
- **SGLang**: Supported
- **VLLM**: Supported

### Database Versions
- TRTLLM: 0.20.0, 1.0.0rc3 (others available)
- Different versions per system in `src/aiconfigurator/systems/`

## Output Structure

When using `--save_dir`, generates:
```
results/
├── agg/
│   ├── best_config_topn.csv          # Top 5 configurations
│   ├── config.yaml                   # Full config
│   ├── pareto.csv                    # Pareto frontier points
│   └── top1/
│       ├── agg/
│       │   ├── agg_config.yaml       # Framework config
│       │   ├── k8s_deploy.yaml       # Kubernetes manifest
│       │   └── node_0_run.sh         # Deployment script
│       └── generator_config.yaml
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

## Design Patterns Worth Noting

### 1. **Layered Configuration Factory**
- Compose configuration from multiple layers (base → mode-specific → profile → user patch)
- Conditional layer application based on context
- Applied at: `src/aiconfigurator/sdk/task.py`

### 2. **Strategy Pattern for Backends**
- Abstract backend interface with specific implementations (TRTLLM, SGLang, VLLM)
- Unified inference simulation across backends
- Applied at: `src/aiconfigurator/sdk/backends/`

### 3. **Operation-Based Performance Modeling**
- Break down inference into composable operations (GEMM, Attention, AllReduce, etc.)
- Query pre-collected CSV databases for operation timings
- Interpolate for unseen configurations
- Applied at: `src/aiconfigurator/sdk/operations.py`

### 4. **Pareto Frontier Analysis**
- Multi-objective optimization (throughput vs latency)
- Constraint satisfaction with exponential penalties
- Applied at: `src/aiconfigurator/sdk/pareto_analysis.py`

## Unique Features

1. **Disaggregated Serving Modeling** - First to model prefill/decode separation
2. **Heterogeneous Deployments** - Different hardware for prefill vs decode
3. **Fine-Grained Quantization** - Per-operation quantization control
4. **Replica-Based Scaling** - Treats xPyD units as scalable replicas
5. **Constraint-Aware Scoring** - Exponential penalty functions for SLA violations

## Integration with Inference Autotuner

### Most Applicable Features
1. **Constraint Handling** - aiconfigurator's SLO model could be adapted for general metrics
2. **Pareto Analysis** - Multi-objective optimization framework
3. **Layered Configuration** - Configuration composition pattern
4. **Backend Strategy** - Abstract interface for different inference engines
5. **Performance Estimation** - Operation-based modeling approach

### Potential Code Reuse
- SLO scoring functions
- Configuration merging/patching logic
- Backend abstraction base classes
- Result aggregation and visualization

## Performance Modeling Approach

1. **Data Collection**: Profile operation timings on target GPU (GEMM, Attention, AllReduce, etc.)
2. **Database Storage**: CSV files organized as `systems/{gpu}/{framework}/{version}/{operation}.txt`
3. **Query & Interpolate**: Look up operation time, interpolate if needed, compose into end-to-end estimates
4. **Latency Correction**: Apply empirical correction factors (default 1.0)
5. **SLA Evaluation**: Check constraints and compute penalty-adjusted score

## Example Outputs

### CLI Default Mode Output
```
Input: QWEN3_32B, 32 GPUs, h200_sxm, TTFT<300ms, TPOT<10ms

Best Configuration: disagg at 913.82 tokens/s/gpu (1.43x better)

Performance Metrics:
- Throughput: 913.82 tokens/s/gpu
- User Throughput: 123.92 tokens/s/user
- TTFT: 202.65 ms (under 300ms SLA ✓)
- TPOT: 8.07 ms (under 10ms SLA ✓)

Deployment: 4 replicas × (4 prefill + 1 decode) = 32 GPUs
```

## File Structure Summary

| Directory | Purpose |
|-----------|---------|
| `src/aiconfigurator/sdk/` | Core optimization engine |
| `src/aiconfigurator/cli/` | CLI interface |
| `src/aiconfigurator/webapp/` | Gradio web UI |
| `src/aiconfigurator/generator/` | Code generation for frameworks |
| `src/aiconfigurator/eval/` | End-to-end pipeline |
| `collector/` | Performance data collection |
| `docs/` | User guides and tutorials |
| `tests/` | Test suite |
| `tools/automation/` | End-to-end automation scripts |

## Resources

- **Main README**: `/root/work/inference-autotuner/third_party/aiconfigurator/README.md`
- **CLI Guide**: `docs/cli_user_guide.md`
- **Advanced Tuning**: `docs/advanced_tuning.md`
- **Deployment Guide**: `docs/dynamo_deployment_guide.md`
- **Developer Guide**: `DEVELOPMENT.md`
- **Detailed Analysis**: `/root/work/inference-autotuner/AICONFIGURATOR_ANALYSIS.md`

## Technology Stack

- **Language**: Python 3.9+
- **Optimization**: NumPy, SciPy (interpolation)
- **Data**: Pandas
- **UI**: Gradio (web), Terminal (CLI)
- **Config**: YAML, Pydantic
- **Templating**: Jinja2
- **Linting**: Ruff
- **Testing**: Pytest

## Key Takeaway

aiconfigurator is essentially a **performance prediction + configuration search system** that:
1. Models LLM inference using operation-level data
2. Searches configuration space automatically
3. Respects SLA constraints with penalty functions
4. Generates ready-to-deploy framework configs
5. Innovates with disaggregated serving modeling

Its most compelling innovation is **separating prefill and decode into independent worker pools**, enabling significantly better throughput under latency constraints.
