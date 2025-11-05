# Layered Config Factory Integration Guide

## Overview

This document describes how to integrate the **Layered Config Factory** feature from aiconfigurator into the inference-autotuner project.

Layered Config Factory is a hierarchical configuration management pattern that allows building final configuration objects through the combination of multiple configuration layers, supporting conditional application, dynamic resolution, and deep merging.

## Analysis of aiconfigurator's Layered Config Factory Architecture

### Core Components

#### 1. ConfigLayer

```python
@dataclass(frozen=True)
class ConfigLayer:
    name: str                                          # Layer name
    data: dict | Callable[[TaskContext], dict]         # Static dict or dynamic function
    condition: Callable[[TaskContext], bool] | None    # Application condition (optional)
```

**Features**:
- **Conditional Application**: Layer only applies when `condition(ctx)` returns True
- **Dynamic Resolution**: `data` can be a function that generates config based on context
- **Immutability**: Uses `frozen=True` to ensure layer definitions are immutable

#### 2. TaskContext

```python
@dataclass
class TaskContext:
    serving_mode: Literal["agg", "disagg"]    # Serving mode
    model_name: str                           # Model name
    system_name: str                          # Hardware system
    backend_name: str                         # Inference backend
    backend_version: str | None               # Backend version
    isl: int                                  # Input sequence length
    osl: int                                  # Output sequence length
    ttft: float                               # Time to first token target
    tpot: float                               # Time per output token target
    total_gpus: int | None                    # GPU count limit
    profiles: list[str]                       # Activated config presets
    yaml_patch: dict                          # User custom patch
    yaml_mode: Literal["patch", "replace"]    # Patch mode
```

#### 3. TaskConfigFactory

```python
class TaskConfigFactory:
    PROFILE_REGISTRY: ClassVar[dict[str, list[ConfigLayer]]] = {}

    @classmethod
    def create(cls, ctx: TaskContext) -> tuple[DefaultMunch, list[str]]:
        # 1. Apply base layers
        # 2. Apply mode layers (agg/disagg)
        # 3. Apply user-defined profiles
        # 4. Apply user patches (yaml_patch)
        # 5. Finalize configuration
        return config, applied_layers
```

### Configuration Layer Application Order

```
1. base-common (Base common configuration)
   ├── model_name, serving_mode
   ├── nextn (automatically set based on model)
   └── runtime_config (isl, osl, ttft, tpot)

2. mode-specific layers
   ├── agg-defaults
   │   └── worker_config (single-node configuration)
   └── disagg-defaults
       ├── prefill_worker_config
       ├── decode_worker_config
       ├── replica_config
       └── advanced_tuning_config

3. profiles (User-activated preset configurations)
   └── Loaded from PROFILE_REGISTRY

4. yaml_patch (User custom patch)
   ├── patch mode: Deep merge into existing config
   └── replace mode: Complete config replacement

5. finalization
   ├── Limit available configs based on total_gpus
   └── Auto-apply quantization modes
```

### Deep Merge Mechanism

```python
def _deep_merge(target: dict, source: Mapping, *, allow_new: bool = True) -> dict:
    for key, value in source.items():
        if key not in target:
            if not allow_new:
                continue
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_merge(target[key], value, allow_new=allow_new)  # Recursive merge
        else:
            target[key] = copy.deepcopy(value)  # Override
```

**Features**:
- Nested dict recursive merging
- Uses deep copy to avoid reference sharing
- `allow_new` parameter controls whether to allow adding new keys (for user patch validation)

## Implementation Plan for inference-autotuner Integration

### Phase 1: Core Architecture Introduction

#### 1.1 Create Configuration Layer Infrastructure

**File**: `src/config/layers.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ClassVar
from collections.abc import Mapping
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigLayer:
    """Configuration layer definition"""
    name: str
    data: dict | Callable[[TaskContext], dict]
    condition: Callable[[TaskContext], bool] | None = None

    def applies_to(self, ctx: TaskContext) -> bool:
        if self.condition is None:
            return True
        try:
            return self.condition(ctx)
        except Exception:
            logger.debug(f"Layer {self.name} condition failed")
            return False

    def resolve(self, ctx: TaskContext) -> dict:
        payload = self.data(ctx) if callable(self.data) else self.data
        return copy.deepcopy(payload)


@dataclass
class TaskContext:
    """Task execution context"""
    # Basic configuration
    model_name: str
    base_runtime: str                           # sglang, vllm, trtllm
    deployment_mode: str                        # docker, ome

    # Runtime parameters
    benchmark_task: str                         # text-to-text, etc.
    traffic_scenarios: list[str]
    num_concurrency: list[int]

    # Optimization objectives
    optimization_strategy: str                  # grid_search, bayesian
    optimization_objective: str                 # minimize_latency, etc.

    # SLO configuration (optional)
    slo_config: dict | None = None

    # Advanced options
    profiles: list[str] = field(default_factory=list)
    user_overrides: dict = field(default_factory=dict)
    override_mode: Literal["patch", "replace"] = "patch"

    # Environment configuration
    gpu_type: str | None = None
    total_gpus: int | None = None


def _deep_merge(target: dict, source: Mapping, *, allow_new: bool = True) -> dict:
    """Deep merge two dictionaries"""
    for key, value in source.items():
        if key not in target:
            if not allow_new:
                continue
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_merge(target[key], value, allow_new=allow_new)
        else:
            target[key] = copy.deepcopy(value)
    return target
```

#### 1.2 Create Configuration Factory

**File**: `src/config/factory.py`

```python
from typing import ClassVar
import copy
import logging
from .layers import ConfigLayer, TaskContext, _deep_merge

logger = logging.getLogger(__name__)


class TaskConfigFactory:
    """Layered configuration factory"""

    PROFILE_REGISTRY: ClassVar[dict[str, list[ConfigLayer]]] = {}

    @classmethod
    def register_profile(cls, name: str, layers: list[ConfigLayer]) -> None:
        """Register configuration preset"""
        cls.PROFILE_REGISTRY[name] = layers
        logger.info(f"Registered profile: {name}")

    @classmethod
    def create(cls, ctx: TaskContext) -> tuple[dict, list[str]]:
        """
        Create configuration from context

        Returns:
            (config_dict, applied_layers): Configuration dict and list of applied layers
        """
        config_dict: dict = {}
        applied_layers: list[str] = []

        # 1. Apply base layers
        for layer in cls._base_layers():
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 2. Apply deployment mode layers
        for layer in cls._deployment_mode_layers(ctx):
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 3. Apply runtime layers
        for layer in cls._runtime_layers(ctx):
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 4. Apply user presets
        for profile in ctx.profiles:
            layers = cls.PROFILE_REGISTRY.get(profile)
            if not layers:
                logger.warning(f"Profile '{profile}' not found, skipping")
                continue
            for layer in layers:
                if layer.applies_to(ctx):
                    _deep_merge(config_dict, layer.resolve(ctx))
                    applied_layers.append(f"profile:{profile}:{layer.name}")

        # 5. Apply user overrides
        if ctx.user_overrides:
            if ctx.override_mode == "replace":
                config_dict = copy.deepcopy(ctx.user_overrides)
                applied_layers.append("user_replace")
            else:
                _deep_merge(config_dict, ctx.user_overrides, allow_new=True)
                applied_layers.append("user_patch")

        # 6. Finalization
        cls._finalize(config_dict, ctx)

        logger.info(f"Applied layers: {applied_layers}")
        return config_dict, applied_layers

    @classmethod
    def _base_layers(cls) -> list[ConfigLayer]:
        """Base configuration layers"""
        return [
            ConfigLayer("base-model", cls._base_model_layer),
            ConfigLayer("base-optimization", cls._base_optimization_layer),
        ]

    @classmethod
    def _deployment_mode_layers(cls, ctx: TaskContext) -> list[ConfigLayer]:
        """Deployment mode layers"""
        if ctx.deployment_mode == "docker":
            return [ConfigLayer("docker-defaults", cls._docker_defaults_layer)]
        elif ctx.deployment_mode == "ome":
            return [ConfigLayer("ome-defaults", cls._ome_defaults_layer)]
        return []

    @classmethod
    def _runtime_layers(cls, ctx: TaskContext) -> list[ConfigLayer]:
        """Runtime layers"""
        if ctx.base_runtime == "sglang":
            return [ConfigLayer("sglang-defaults", cls._sglang_defaults_layer)]
        elif ctx.base_runtime == "vllm":
            return [ConfigLayer("vllm-defaults", cls._vllm_defaults_layer)]
        elif ctx.base_runtime == "trtllm":
            return [ConfigLayer("trtllm-defaults", cls._trtllm_defaults_layer)]
        return []

    @staticmethod
    def _base_model_layer(ctx: TaskContext) -> dict:
        """Base model configuration"""
        return {
            "task_name": f"{ctx.model_name}_{ctx.optimization_strategy}",
            "model": {
                "id_or_path": ctx.model_name,
                "namespace": "autotuner"
            },
            "base_runtime": ctx.base_runtime,
        }

    @staticmethod
    def _base_optimization_layer(ctx: TaskContext) -> dict:
        """Base optimization configuration"""
        return {
            "optimization": {
                "strategy": ctx.optimization_strategy,
                "objective": ctx.optimization_objective,
                "max_iterations": 10,
                "timeout_per_iteration": 600
            },
            "benchmark": {
                "task": ctx.benchmark_task,
                "model_name": ctx.model_name,
                "traffic_scenarios": ctx.traffic_scenarios,
                "num_concurrency": ctx.num_concurrency,
            }
        }

    @staticmethod
    def _docker_defaults_layer(ctx: TaskContext) -> dict:
        """Docker mode default configuration"""
        return {
            "runtime_image_tag": "v0.5.2-cu126",
            "parameters": {
                "tp-size": [1, 2, 4],
                "mem-fraction-static": [0.85, 0.9]
            }
        }

    @staticmethod
    def _ome_defaults_layer(ctx: TaskContext) -> dict:
        """OME mode default configuration"""
        return {
            "parameters": {
                "tp-size": [1, 2, 4, 8],
                "mem-fraction-static": [0.8, 0.85, 0.9]
            }
        }

    @staticmethod
    def _sglang_defaults_layer(ctx: TaskContext) -> dict:
        """SGLang runtime default parameters"""
        return {
            "parameters": {
                "schedule-policy": ["lpm"],
                "enable-torch-compile": [True, False]
            }
        }

    @staticmethod
    def _vllm_defaults_layer(ctx: TaskContext) -> dict:
        """vLLM runtime default parameters"""
        return {
            "parameters": {
                "enable-chunked-prefill": [True, False]
            }
        }

    @staticmethod
    def _trtllm_defaults_layer(ctx: TaskContext) -> dict:
        """TensorRT-LLM runtime default parameters"""
        return {
            "parameters": {
                "max-batch-size": [32, 64, 128]
            }
        }

    @classmethod
    def _finalize(cls, config: dict, ctx: TaskContext) -> None:
        """Finalize configuration"""
        # Apply SLO configuration
        if ctx.slo_config:
            config["slo"] = ctx.slo_config

        # Limit GPU count
        if ctx.total_gpus and "parameters" in config:
            if "tp-size" in config["parameters"]:
                config["parameters"]["tp-size"] = [
                    tp for tp in config["parameters"]["tp-size"]
                    if tp <= ctx.total_gpus
                ]
```

### Phase 2: Profiles System

#### 2.1 Define Common Presets

**File**: `src/config/profiles.py`

```python
from .layers import ConfigLayer
from .factory import TaskConfigFactory


# High throughput preset
HIGH_THROUGHPUT_LAYERS = [
    ConfigLayer(
        name="high-throughput-params",
        data={
            "optimization": {
                "objective": "maximize_throughput",
                "max_iterations": 20
            },
            "parameters": {
                "mem-fraction-static": [0.95],
                "tp-size": [4, 8],
            },
            "benchmark": {
                "num_concurrency": [16, 32, 64]
            }
        }
    )
]

# Low latency preset
LOW_LATENCY_LAYERS = [
    ConfigLayer(
        name="low-latency-params",
        data={
            "optimization": {
                "objective": "minimize_latency",
                "max_iterations": 15
            },
            "parameters": {
                "mem-fraction-static": [0.7, 0.8],
                "tp-size": [1, 2],
            },
            "benchmark": {
                "num_concurrency": [1, 4, 8]
            }
        }
    )
]

# Quick test preset
QUICK_TEST_LAYERS = [
    ConfigLayer(
        name="quick-test-params",
        data={
            "optimization": {
                "max_iterations": 2
            },
            "parameters": {
                "tp-size": [1],
                "mem-fraction-static": [0.85]
            },
            "benchmark": {
                "num_concurrency": [1],
                "traffic_scenarios": ["D(100,100)"]
            }
        }
    )
]

# Production preset
PRODUCTION_LAYERS = [
    ConfigLayer(
        name="production-params",
        data={
            "optimization": {
                "max_iterations": 30,
                "timeout_per_iteration": 900
            },
            "slo": {
                "ttft": {"threshold": 1.0, "weight": 2.0},
                "tpot": {"threshold": 0.05, "weight": 2.0},
                "latency": {
                    "p90": {
                        "threshold": 5.0,
                        "weight": 3.0,
                        "hard_fail": True,
                        "fail_ratio": 0.2
                    }
                }
            }
        }
    )
]


def register_builtin_profiles():
    """Register built-in presets"""
    TaskConfigFactory.register_profile("high-throughput", HIGH_THROUGHPUT_LAYERS)
    TaskConfigFactory.register_profile("low-latency", LOW_LATENCY_LAYERS)
    TaskConfigFactory.register_profile("quick-test", QUICK_TEST_LAYERS)
    TaskConfigFactory.register_profile("production", PRODUCTION_LAYERS)
```

### Phase 3: Integration with Existing System

#### 3.1 Modify Task Creation Flow

**File**: `src/web/routes/tasks.py` (add the following)

```python
from src.config.factory import TaskConfigFactory
from src.config.layers import TaskContext
from src.config.profiles import register_builtin_profiles

# Register presets at application startup
register_builtin_profiles()


@router.post("/tasks/from-context")
async def create_task_from_context(
    context_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """Create task from context (using layered configuration)"""

    # Build context
    ctx = TaskContext(
        model_name=context_data["model_name"],
        base_runtime=context_data["base_runtime"],
        deployment_mode=context_data.get("deployment_mode", "docker"),
        benchmark_task=context_data.get("benchmark_task", "text-to-text"),
        traffic_scenarios=context_data.get("traffic_scenarios", ["D(100,100)"]),
        num_concurrency=context_data.get("num_concurrency", [1, 4]),
        optimization_strategy=context_data.get("optimization_strategy", "grid_search"),
        optimization_objective=context_data.get("optimization_objective", "minimize_latency"),
        profiles=context_data.get("profiles", []),
        user_overrides=context_data.get("overrides", {}),
        slo_config=context_data.get("slo"),
        total_gpus=context_data.get("total_gpus")
    )

    # Generate configuration
    config_dict, applied_layers = TaskConfigFactory.create(ctx)

    # Create task (using existing logic)
    task = Task(
        task_name=config_dict["task_name"],
        config=config_dict,
        status=TaskStatus.PENDING,
        metadata={"applied_layers": applied_layers}
    )

    db.add(task)
    await db.commit()
    await db.refresh(task)

    return {
        "task": task,
        "applied_layers": applied_layers
    }
```

#### 3.2 Frontend Integration

**File**: `frontend/src/pages/NewTask.tsx` (add profile selector)

```typescript
// Profile selector component
const ProfileSelector: React.FC = () => {
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>([]);

  const profiles = [
    { id: "high-throughput", name: "High Throughput", description: "Maximize throughput" },
    { id: "low-latency", name: "Low Latency", description: "Minimize latency" },
    { id: "quick-test", name: "Quick Test", description: "2 iterations for fast validation" },
    { id: "production", name: "Production", description: "Full test with SLO constraints" }
  ];

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Configuration Presets</label>
      <div className="grid grid-cols-2 gap-2">
        {profiles.map(profile => (
          <button
            key={profile.id}
            onClick={() => toggleProfile(profile.id)}
            className={`p-3 border rounded ${
              selectedProfiles.includes(profile.id)
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300'
            }`}
          >
            <div className="font-semibold">{profile.name}</div>
            <div className="text-xs text-gray-500">{profile.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
};
```

### Phase 4: Advanced Features

#### 4.1 Dynamic Configuration Layers

```python
# Auto-adjust parameters based on GPU type
def gpu_aware_layer(ctx: TaskContext) -> dict:
    if ctx.gpu_type == "H100":
        return {
            "parameters": {
                "tp-size": [1, 2, 4, 8],
                "mem-fraction-static": [0.85, 0.9, 0.95]
            }
        }
    elif ctx.gpu_type == "A100":
        return {
            "parameters": {
                "tp-size": [1, 2, 4],
                "mem-fraction-static": [0.8, 0.85]
            }
        }
    return {}

# Register conditional layer
ConfigLayer(
    name="gpu-aware",
    data=gpu_aware_layer,
    condition=lambda ctx: ctx.gpu_type is not None
)
```

#### 4.2 Configuration Validation

```python
class ConfigValidator:
    @staticmethod
    def validate(config: dict) -> list[str]:
        """Validate configuration and return warning list"""
        warnings = []

        # Check parameter ranges
        if "parameters" in config:
            if "tp-size" in config["parameters"]:
                tp_sizes = config["parameters"]["tp-size"]
                if max(tp_sizes) > 8:
                    warnings.append("tp-size > 8 may require multiple nodes")

        # Check SLO consistency
        if "slo" in config:
            if "ttft" in config["slo"] and "tpot" in config["slo"]:
                if config["slo"]["ttft"]["threshold"] < config["slo"]["tpot"]["threshold"]:
                    warnings.append("TTFT threshold should be larger than TPOT")

        return warnings
```

## Implementation Steps Summary

### Step 1: Create Core Files
```bash
# Create configuration management directory
mkdir -p src/config

# Create core files
touch src/config/__init__.py
touch src/config/layers.py       # Configuration layer definitions
touch src/config/factory.py      # Configuration factory
touch src/config/profiles.py     # Preset configurations
touch src/config/validator.py    # Configuration validation
```

### Step 2: Install Dependencies
```bash
# No extra dependencies needed, using standard library
# If you need better dictionary access syntax, optionally install:
pip install munch
```

### Step 3: Backend Integration
1. Register presets at startup in `src/web/app.py`
2. Add new task creation endpoint in `src/web/routes/tasks.py`
3. Keep existing endpoints backward compatible

### Step 4: Frontend Integration
1. Add profile selector in `frontend/src/pages/NewTask.tsx`
2. Add new API calls in `frontend/src/services/api.ts`
3. Update type definitions in `frontend/src/types/index.ts`

### Step 5: Testing & Documentation
1. Write unit tests in `tests/test_config_factory.py`
2. Write integration tests to verify end-to-end flow
3. Update `README.md` and `CLAUDE.md` documentation

## Benefits Analysis

### 1. Code Maintainability
- **Before**: Adding new configuration options requires modifying multiple code locations
- **After**: Only need to add new configuration layers, other code unchanged

### 2. User Experience
- **Before**: Users need to manually fill in all parameters
- **After**: Select presets to quickly start, advanced users can customize

### 3. Configuration Reuse
- **Before**: Similar scenarios require repeated configuration
- **After**: Achieve configuration reuse through presets and layer composition

### 4. Extensibility
- **Before**: Adding new deployment modes or runtimes requires extensive code changes
- **After**: Only need to add corresponding configuration layers

## Example: Creating Tasks with Layered Configuration

```python
# Python API
from src.config.factory import TaskConfigFactory
from src.config.layers import TaskContext

ctx = TaskContext(
    model_name="llama-3-2-1b-instruct",
    base_runtime="sglang",
    deployment_mode="docker",
    benchmark_task="text-to-text",
    traffic_scenarios=["D(100,100)"],
    num_concurrency=[1, 4, 8],
    optimization_strategy="grid_search",
    optimization_objective="minimize_latency",
    profiles=["low-latency", "production"],  # Combine multiple presets
    user_overrides={
        "parameters": {
            "tp-size": [2]  # User override
        }
    }
)

config, layers = TaskConfigFactory.create(ctx)
print(f"Applied layers: {layers}")
# Output: ['base-model', 'base-optimization', 'docker-defaults',
#          'sglang-defaults', 'profile:low-latency:low-latency-params',
#          'profile:production:production-params', 'user_patch']
```

```bash
# REST API
curl -X POST http://localhost:8000/api/tasks/from-context \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama-3-2-1b-instruct",
    "base_runtime": "sglang",
    "deployment_mode": "docker",
    "profiles": ["low-latency"],
    "overrides": {
      "parameters": {
        "tp-size": [1]
      }
    }
  }'
```

## References

- aiconfigurator source code: `third_party/aiconfigurator/src/aiconfigurator/sdk/task.py`
- Design patterns: Strategy Pattern + Factory Pattern + Chain of Responsibility
- Similar implementations: Hydra (Facebook), OmegaConf (PyTorch Lightning)
