# Configuration Profiles Guide

Configuration profiles are pre-defined configurations for common use cases in the inference autotuner. They allow you to quickly create tasks without manually specifying all parameters.

## Overview

Profiles use the **Layered Config Factory** system to build task configurations by merging:
1. Base layers (model, optimization)
2. Deployment mode layers (Docker, OME)
3. Runtime layers (SGLang, vLLM, TensorRT-LLM)
4. **Profile layers** (your selected profile)
5. User overrides (optional customization)

## Available Profiles

### 1. Quick Test (`quick-test`)

**Purpose**: Fast validation with minimal experiments for development and testing

**Use Cases**:
- Development and debugging
- Smoke testing
- CI/CD pipelines
- Quick configuration validation

**Configuration**:
- Iterations: 2
- TP size: [1]
- Memory fraction: [0.85]
- Concurrency: [1]
- Traffic: D(100,100)

**Example**:
```bash
curl -X POST http://localhost:8000/api/tasks/from-context \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama-3-2-1b-instruct",
    "base_runtime": "sglang",
    "deployment_mode": "docker",
    "profiles": ["quick-test"]
  }'
```

---

### 2. Balanced (`balanced`)

**Purpose**: Balanced configuration exploring common parameter ranges

**Use Cases**:
- General purpose testing
- Exploring parameter space
- Baseline performance measurement
- First-time model tuning

**Configuration**:
- Iterations: 15
- TP size: [1, 2, 4]
- Memory fraction: [0.8, 0.85, 0.9]
- Concurrency: [4, 8, 16]
- Timeout: 600s per iteration

**Recommended for**: Users starting without specific requirements

---

### 3. Low Latency (`low-latency`)

**Purpose**: Minimize response latency for real-time interactive workloads

**Use Cases**:
- Chatbots and conversational AI
- Real-time applications
- Interactive services
- Single-user scenarios

**Configuration**:
- Objective: minimize_latency
- Iterations: 15
- TP size: [1, 2] (fewer GPUs for lower latency)
- Memory fraction: [0.7, 0.8] (more headroom)
- Concurrency: [1, 4, 8] (lower concurrency)

**Recommended for**: Production interactive services, single-GPU setups

---

### 4. High Throughput (`high-throughput`)

**Purpose**: Maximize request throughput with high concurrency testing

**Use Cases**:
- Batch processing workloads
- High-traffic scenarios
- Multi-user concurrent requests
- Maximum utilization testing

**Configuration**:
- Objective: maximize_throughput
- Iterations: 20
- TP size: [4, 8] (more GPUs for parallelism)
- Memory fraction: [0.95] (maximize memory usage)
- Concurrency: [16, 32, 64] (high concurrency)

**Recommended for**: Production high-traffic deployments, multi-GPU systems

---

### 5. Cost Optimization (`cost-optimization`)

**Purpose**: Balance performance with minimal resource usage and cost

**Use Cases**:
- Budget-conscious deployments
- Development environments
- Startups and small teams
- Resource-constrained scenarios

**Configuration**:
- Objective: minimize_latency (efficiency focus)
- Iterations: 12
- TP size: [1, 2] (fewer GPUs to save cost)
- Memory fraction: [0.75, 0.85] (lower memory usage)
- Concurrency: [4, 8, 16] (moderate concurrency)

**Recommended for**: Development, startups, single-GPU deployments

---

### 6. Production (`production`)

**Purpose**: Comprehensive testing with SLO constraints for production deployment

**Use Cases**:
- Production readiness testing
- SLO validation
- Performance guarantee verification
- Pre-deployment validation

**Configuration**:
- Iterations: 30 (comprehensive)
- Timeout: 900s per iteration
- **SLO Constraints**:
  - TTFT: ≤ 1.0s (weight: 2.0)
  - TPOT: ≤ 0.05s (weight: 2.0)
  - P90 Latency: ≤ 5.0s (weight: 3.0, hard_fail at >20% violation)

**Recommended for**: Production deployment, SLO-critical services

---

## Using Profiles

### API Endpoints

#### List All Profiles
```bash
GET /api/profiles/
```

**Response**:
```json
[
  {
    "name": "balanced",
    "description": "Balanced configuration exploring common parameter ranges",
    "use_case": "General purpose testing, exploring parameter space",
    "tags": ["balanced", "general", "exploration"],
    "recommended_for": ["general", "exploration", "baseline"],
    "layers_count": 1
  },
  ...
]
```

#### Get Profile Details
```bash
GET /api/profiles/{profile_name}
```

**Example**:
```bash
curl http://localhost:8000/api/profiles/balanced
```

**Response**:
```json
{
  "name": "balanced",
  "description": "Balanced configuration exploring common parameter ranges",
  "use_case": "General purpose testing, exploring parameter space",
  "tags": ["balanced", "general", "exploration"],
  "recommended_for": ["general", "exploration", "baseline"],
  "layers_count": 1,
  "layer_names": ["balanced-params"]
}
```

### Create Task from Profile

#### Basic Usage
```bash
POST /api/tasks/from-context

{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "deployment_mode": "docker",
  "profiles": ["quick-test"]
}
```

#### Multiple Profiles
```bash
{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "deployment_mode": "docker",
  "profiles": ["low-latency", "production"]  # Combines both profiles
}
```

#### With Custom Overrides
```bash
{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "deployment_mode": "docker",
  "profiles": ["balanced"],
  "user_overrides": {
    "optimization": {
      "max_iterations": 5  # Override default from profile
    }
  }
}
```

## Profile Selection Guide

| Scenario | Recommended Profile | Rationale |
|----------|-------------------|-----------|
| First-time tuning | `balanced` | Explores common ranges without bias |
| Quick validation | `quick-test` | Fast feedback, minimal resources |
| Chatbot deployment | `low-latency` | Optimizes for user experience |
| Batch inference | `high-throughput` | Maximizes concurrent processing |
| Limited budget | `cost-optimization` | Minimizes GPU count and memory |
| Production launch | `production` | Validates SLO compliance |
| Development/testing | `quick-test` or `cost-optimization` | Fast iteration, low cost |
| Multi-GPU cluster | `high-throughput` | Leverages available resources |
| Single-GPU machine | `low-latency` or `cost-optimization` | Optimized for constraints |

## Advanced Usage

### Custom GPU Constraints
```bash
{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "profiles": ["high-throughput"],
  "total_gpus": 2  # Limits tp-size to ≤2
}
```

### Custom SLO Configuration
```bash
{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "profiles": ["balanced"],
  "slo_config": {
    "ttft": {"threshold": 0.5, "weight": 3.0},
    "latency": {
      "p90": {"threshold": 3.0, "weight": 2.0, "hard_fail": true}
    }
  }
}
```

### Override Mode
```bash
{
  "model_name": "llama-3-2-1b-instruct",
  "base_runtime": "sglang",
  "profiles": ["balanced"],
  "user_overrides": {...},
  "override_mode": "patch"  # or "replace"
}
```

- **patch** (default): Merges overrides with profile config
- **replace**: Completely replaces config with overrides

## Creating Custom Profiles

To create custom profiles, add them to `src/config/profiles.py`:

```python
# Custom profile example
CUSTOM_LAYERS = [
    ConfigLayer(
        name="my-custom-params",
        data={
            "optimization": {
                "objective": "minimize_latency",
                "max_iterations": 10
            },
            "parameters": {
                "tp-size": [2],
                "mem-fraction-static": [0.9]
            }
        }
    )
]

CUSTOM_METADATA = ProfileMetadata(
    name="my-custom",
    description="My custom profile for specific use case",
    use_case="Specific deployment scenario",
    tags=["custom", "specialized"],
    recommended_for=["my-use-case"]
)

# Register in register_builtin_profiles()
TaskConfigFactory.register_profile("my-custom", CUSTOM_LAYERS, CUSTOM_METADATA)
```

## FAQ

**Q: Can I combine multiple profiles?**
A: Yes! Specify multiple profiles in the `profiles` array. They will be applied in order, with later profiles overriding earlier ones.

**Q: How do user_overrides interact with profiles?**
A: By default (`override_mode: "patch"`), user overrides are merged with the profile config. Nested dictionaries are deep-merged.

**Q: What happens if a profile doesn't exist?**
A: The system logs a warning and skips that profile. Other profiles and base layers still apply.

**Q: How can I see what configuration a profile generates?**
A: Use the `POST /api/tasks/from-context` endpoint and check the `generated_config` field in the response.

**Q: Can I use profiles with the old `/api/tasks/` endpoint?**
A: No, profiles only work with `/api/tasks/from-context`. The old endpoint requires explicit configuration.

## See Also

- [Layered Config Integration Guide](./LAYERED_CONFIG_INTEGRATION.md) - Architecture details
- [SLO Scoring Documentation](./SLO_SCORING.md) - SLO configuration guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
