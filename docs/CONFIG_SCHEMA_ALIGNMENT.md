# Configuration Schema Alignment Design

本文档定义 inference-autotuner 和 aiconfigurator 之间的配置结构对齐方案。

## 设计原则

1. **向后兼容**: 保持现有扁平化 parameters 的兼容性
2. **结构化优先**: 新配置使用分组结构
3. **灵活性**: 支持两种配置格式的互相转换
4. **完整性**: 保留 aiconfigurator 的所有配置信息

## 配置结构设计

### 1. System Configuration (系统配置)

```typescript
interface SystemConfig {
  gpu_type: string;           // "h200_sxm", "h100_pcie", "a100_sxm", etc.
  total_gpus: number;         // Total number of GPUs available
  memory_per_gpu?: number;    // GB per GPU
  nvlink_bandwidth?: number;  // GB/s
  pcie_bandwidth?: number;    // GB/s
  interconnect_type?: string; // "nvlink", "pcie", "infiniband"
}
```

**Aiconfigurator mapping**:
```yaml
# Aiconfigurator format
system_name: "h200_sxm"
total_gpus: 8

# Maps to our SystemConfig
system_config:
  gpu_type: "h200_sxm"
  total_gpus: 8
  memory_per_gpu: 141  # From system profile
  nvlink_bandwidth: 900
```

### 2. Parallel Configuration (并行配置)

```typescript
interface ParallelConfig {
  tp: number;              // Tensor Parallelism
  pp?: number;             // Pipeline Parallelism (default: 1)
  dp?: number;             // Data Parallelism (default: 1)
  moe_tp?: number;         // MoE Tensor Parallelism (for MoE models)
  moe_ep?: number;         // MoE Expert Parallelism (for MoE models)
}
```

**Aiconfigurator mapping**:
```yaml
# Aiconfigurator format
worker_config:
  tp: 4
  pp: 2
  dp: 1
  moe_tp: 2

# Maps to our ParallelConfig
parallel_config:
  tp: 4
  pp: 2
  dp: 1
  moe_tp: 2
```

**Runtime parameters mapping**:
```json
{
  "parallel_config": {
    "tp": 4,
    "pp": 2
  },

  // Converts to runtime flags:
  "parameters": {
    "tp-size": [4],
    "pp-size": [2]  // If runtime supports it
  }
}
```

### 3. Quantization Configuration (量化配置)

```typescript
interface QuantizationConfig {
  gemm_quant_mode?: string;      // "float16", "fp8", "int8", "int4"
  kvcache_quant_mode?: string;   // "float16", "fp8", "int8"
  fmha_quant_mode?: string;      // FlashAttention quantization
  activation_quant_mode?: string;
  weight_quant_mode?: string;
}
```

**Aiconfigurator mapping**:
```yaml
# Aiconfigurator format
worker_config:
  gemm_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"

# Maps to our QuantizationConfig
quantization_config:
  gemm_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"
```

**Runtime parameters mapping**:
```json
{
  "quantization_config": {
    "gemm_quant_mode": "fp8",
    "kvcache_quant_mode": "fp8"
  },

  // Converts to runtime flags (SGLang example):
  "parameters": {
    "kv-cache-dtype": ["fp8"],
    "quantization": ["fp8"]  // If supported
  }
}
```

### 4. Memory Configuration (内存配置)

```typescript
interface MemoryConfig {
  mem_fraction_static?: number;   // Static memory fraction (0.0-1.0)
  gpu_memory_utilization?: number; // vLLM-style memory util (0.0-1.0)
  max_model_len?: number;          // Maximum sequence length
  enforce_eager?: boolean;         // Disable CUDA graphs
}
```

**Usage**:
```json
{
  "memory_config": {
    "mem_fraction_static": 0.85,
    "max_model_len": 8192
  },

  // Converts to runtime flags:
  "parameters": {
    "mem-fraction-static": [0.85],
    "max-model-len": [8192]
  }
}
```

### 5. Scheduling Configuration (调度配置)

```typescript
interface SchedulingConfig {
  schedule_policy?: string;       // "lpm", "fcfs", "priority"
  max_num_batched_tokens?: number;
  max_num_seqs?: number;
  chunked_prefill_enabled?: boolean;
}
```

### 6. Advanced Tuning Configuration (高级调优)

```typescript
interface AdvancedTuningConfig {
  enable_chunked_prefill?: boolean;
  max_batch_size?: number;
  max_prefill_batch_size?: number;
  enable_prefix_caching?: boolean;
  disable_sliding_window?: boolean;
  swap_space?: number;  // GB
}
```

## 完整任务配置结构

### 新的 TaskConfig 结构

```typescript
interface TaskConfig {
  // Basic info
  task_name: string;
  description?: string;
  deployment_mode: "docker" | "ome";

  // Model info
  model: {
    id_or_path: string;
    namespace: string;
  };

  // Runtime
  base_runtime: "sglang" | "vllm";
  runtime_image_tag?: string;

  // ===== STRUCTURED CONFIGS =====

  // System configuration (NEW)
  system_config?: SystemConfig;

  // Parallel configuration (NEW)
  parallel_config?: ParallelConfig;

  // Quantization configuration (NEW)
  quantization_config?: QuantizationConfig;

  // Memory configuration (NEW)
  memory_config?: MemoryConfig;

  // Scheduling configuration (NEW)
  scheduling_config?: SchedulingConfig;

  // Advanced tuning (NEW)
  advanced_tuning_config?: AdvancedTuningConfig;

  // ===== LEGACY COMPATIBILITY =====

  // Legacy flat parameters (backward compatible)
  parameters?: Record<string, any[]>;

  // ===== OTHER CONFIGS =====

  optimization: {
    strategy: string;
    objective: string;
    max_iterations: number;
    timeout_per_iteration: number;
  };

  benchmark: {
    task: string;
    model_name: string;
    model_tokenizer: string;
    traffic_scenarios: string[];
    num_concurrency: number[];
    max_time_per_iteration: number;
    max_requests_per_iteration: number;
    additional_params: Record<string, any>;
  };

  slo?: SLOConfig;

  metadata?: {
    aiconfigurator_predictions?: any;
    aiconfigurator_source?: string;
    config_format?: "structured" | "flat";  // NEW: indicates format
  };
}
```

## 配置转换规则

### 从 Aiconfigurator 到 Inference-Autotuner

```python
def convert_aiconfig_to_task(aiconfig: dict) -> TaskConfig:
    """Convert aiconfigurator config to task config."""
    worker_config = aiconfig.get('worker_config', {})

    return {
        # System config
        'system_config': {
            'gpu_type': aiconfig.get('system_name'),
            'total_gpus': worker_config.get('tp', 1) * worker_config.get('pp', 1),
        },

        # Parallel config
        'parallel_config': {
            'tp': worker_config.get('tp', 1),
            'pp': worker_config.get('pp', 1),
            'dp': worker_config.get('dp', 1),
            'moe_tp': worker_config.get('moe_tp'),
            'moe_ep': worker_config.get('moe_ep'),
        },

        # Quantization config
        'quantization_config': {
            'gemm_quant_mode': worker_config.get('gemm_quant_mode'),
            'kvcache_quant_mode': worker_config.get('kvcache_quant_mode'),
            'fmha_quant_mode': worker_config.get('fmha_quant_mode'),
        },

        # Memory config (from advanced_tuning_config if present)
        'memory_config': {
            'mem_fraction_static': aiconfig.get('advanced_tuning_config', {}).get('mem_fraction_static'),
        },

        # Metadata
        'metadata': {
            'aiconfigurator_predictions': aiconfig.get('predicted_metrics'),
            'aiconfigurator_source': 'imported',
            'config_format': 'structured',
        }
    }
```

### 从结构化配置生成运行时参数

```python
def generate_runtime_parameters(config: TaskConfig, runtime: str) -> dict:
    """Generate runtime-specific parameters from structured config."""
    params = {}

    # Parallel config
    if config.get('parallel_config'):
        pc = config['parallel_config']
        params['tp-size'] = [pc['tp']]
        if pc.get('pp') and pc['pp'] > 1:
            params['pp-size'] = [pc['pp']]

    # Quantization config
    if config.get('quantization_config'):
        qc = config['quantization_config']
        if qc.get('kvcache_quant_mode'):
            params['kv-cache-dtype'] = [qc['kvcache_quant_mode']]
        if runtime == 'sglang' and qc.get('gemm_quant_mode') == 'fp8':
            params['quantization'] = ['fp8']

    # Memory config
    if config.get('memory_config'):
        mc = config['memory_config']
        if mc.get('mem_fraction_static'):
            params['mem-fraction-static'] = [mc['mem_fraction_static']]
        if mc.get('max_model_len'):
            params['max-model-len'] = [mc['max_model_len']]

    # Scheduling config
    if config.get('scheduling_config'):
        sc = config['scheduling_config']
        if sc.get('schedule_policy'):
            params['schedule-policy'] = [sc['schedule_policy']]

    return params
```

## 数据库模式更新

### Task Model 更新

```python
class Task(Base):
    # ... existing fields ...

    # New structured config fields (stored as JSON)
    system_config = Column(JSON, nullable=True)
    parallel_config = Column(JSON, nullable=True)
    quantization_config = Column(JSON, nullable=True)
    memory_config = Column(JSON, nullable=True)
    scheduling_config = Column(JSON, nullable=True)
    advanced_tuning_config = Column(JSON, nullable=True)

    # Legacy parameters field (kept for backward compatibility)
    parameters = Column(JSON, nullable=False)
```

## 迁移策略

### Phase 1: 添加新字段（向后兼容）
- ✅ 添加新的结构化配置字段
- ✅ 保留 `parameters` 字段
- ✅ 在创建任务时，如果提供了结构化配置，自动生成 `parameters`
- ✅ 如果只提供了 `parameters`，结构化字段为 null

### Phase 2: 更新 UI（渐进式）
- ✅ 新增"高级配置"tab，显示结构化配置
- ✅ 保留现有参数表单作为"专家模式"
- ✅ 从 aiconfigurator 导入时使用结构化配置

### Phase 3: 更新 Preset 系统
- ✅ Preset 使用结构化配置
- ✅ 转换旧的 preset 格式

## 示例配置

### Aiconfigurator 导入示例

**输入** (aiconfigurator YAML):
```yaml
model_name: "LLAMA_70B"
serving_mode: "agg"
system_name: "h200_sxm"
worker_config:
  tp: 4
  pp: 2
  dp: 1
  gemm_quant_mode: "fp8"
  kvcache_quant_mode: "fp8"
predicted_metrics:
  throughput: 1200.0
  ttft: 150.0
  tpot: 12.0
```

**输出** (Task JSON):
```json
{
  "task_name": "aiconfig-llama_70b-tp4-pp2",
  "system_config": {
    "gpu_type": "h200_sxm",
    "total_gpus": 8
  },
  "parallel_config": {
    "tp": 4,
    "pp": 2,
    "dp": 1
  },
  "quantization_config": {
    "gemm_quant_mode": "fp8",
    "kvcache_quant_mode": "fp8"
  },
  "parameters": {
    "tp-size": [4],
    "kv-cache-dtype": ["fp8"]
  },
  "metadata": {
    "aiconfigurator_predictions": {
      "throughput": 1200.0,
      "ttft": 150.0,
      "tpot": 12.0
    },
    "config_format": "structured"
  }
}
```

## Runtime-Specific Parameter Mapping

### SGLang Parameters
```python
SGLANG_PARAM_MAPPING = {
    'parallel_config': {
        'tp': 'tp-size',
        'dp': 'dp-size',
    },
    'quantization_config': {
        'kvcache_quant_mode': 'kv-cache-dtype',
    },
    'memory_config': {
        'mem_fraction_static': 'mem-fraction-static',
        'max_model_len': 'max-model-len',
    },
    'scheduling_config': {
        'schedule_policy': 'schedule-policy',
    }
}
```

### vLLM Parameters
```python
VLLM_PARAM_MAPPING = {
    'parallel_config': {
        'tp': 'tensor-parallel-size',
        'pp': 'pipeline-parallel-size',
    },
    'quantization_config': {
        'kvcache_quant_mode': 'kv-cache-dtype',
    },
    'memory_config': {
        'gpu_memory_utilization': 'gpu-memory-utilization',
        'max_model_len': 'max-model-len',
    }
}
```

## 验证规则

```python
def validate_structured_config(config: TaskConfig) -> List[str]:
    """Validate structured configuration."""
    errors = []

    # Parallel config validation
    if pc := config.get('parallel_config'):
        if pc['tp'] * pc.get('pp', 1) > config.get('system_config', {}).get('total_gpus', 1):
            errors.append("TP * PP exceeds total available GPUs")

    # Quantization config validation
    if qc := config.get('quantization_config'):
        valid_modes = ['float16', 'fp8', 'int8', 'int4']
        if qc.get('gemm_quant_mode') not in valid_modes:
            errors.append(f"Invalid gemm_quant_mode: {qc['gemm_quant_mode']}")

    return errors
```

## 文档更新清单

- [ ] 更新 CLAUDE.md 添加新配置结构说明
- [ ] 更新 examples/ 中的示例任务
- [ ] 创建配置迁移指南
- [ ] 更新 API 文档
