# AIConfigurator to SGLang Quantization Parameter Mapping

本文档将aiconfigurator的quantization config字段映射到SGLang的对应参数，并提供扩展参数集合建议。

## AIConfigurator Quantization Config 字段概览

aiconfigurator定义了4个主要的量化配置维度（参考：`third_party/aiconfigurator/src/aiconfigurator/sdk/common.py:328-387`）:

1. **gemm_quant_mode** - GEMM矩阵运算量化模式
2. **kvcache_quant_mode** - KV Cache存储量化模式
3. **fmha_quant_mode** - Fused Multi-Head Attention量化模式
4. **moe_quant_mode** - Mixture of Experts量化模式

每个模式都有不同的内存占用、计算加速比和精度特征。

---

## 1. GEMM Quantization Mode

**概念**: GEMM (General Matrix Multiplication) 量化控制模型权重和激活的精度。

### AIConfigurator定义

```python
class GEMMQuantMode(Enum):
    float16     = QuantMapping(memory=2, compute=1, name="float16")    # w16a16 全精度
    int8_wo     = QuantMapping(memory=1, compute=1, name="int8_wo")    # w8a16 仅权重量化
    int4_wo     = QuantMapping(memory=0.5, compute=1, name="int4_wo")  # w4a16 仅权重量化
    fp8         = QuantMapping(memory=1, compute=2, name="fp8")        # w8a8 FP8量化
    sq          = QuantMapping(memory=1, compute=2, name="sq")         # w8a8 INT8量化
    fp8_block   = QuantMapping(memory=1, compute=2, name="fp8_block")  # TensorRT-LLM Torch FP8
    fp8_ootb    = QuantMapping(memory=1, compute=2, name="fp8_ootb")   # TensorRT-LLM TRT backend
    nvfp4       = QuantMapping(memory=0.5, compute=4, name="nvfp4")    # NVIDIA FP4 (Blackwell)
```

**QuantMapping说明**:
- `memory`: 相对FP16的内存占用比例
- `compute`: 相对FP16的计算加速比
- `name`: 量化方法名称

### 映射到SGLang参数

| AIConfigurator | SGLang --quantization | SGLang --dtype | 说明 |
|----------------|----------------------|----------------|------|
| `float16` | (不设置) | `float16` 或 `half` | 默认FP16权重 |
| `int8_wo` | (不适用) | `float16` | SGLang不直接支持INT8 weight-only |
| `int4_wo` | `awq` 或 `gptq` | `half` | 4-bit权重量化，需离线量化模型 |
| `fp8` | `fp8` 或 `w8a8_fp8` | `auto` | 在线FP8量化或离线FP8模型 |
| `sq` | `w8a8_int8` | `auto` | 8-bit INT8权重和激活 |
| `fp8_block` | `modelopt_fp8` | `auto` | NVIDIA ModelOpt FP8量化 |
| `fp8_ootb` | (不适用) | - | TensorRT-LLM专用 |
| `nvfp4` | `modelopt_fp4` 或 `mxfp4` | `auto` | FP4量化 (Blackwell GPU) |

### 扩展建议：SGLang支持但AIConfigurator未定义的GEMM量化方法

```python
# 建议扩展到AIConfigurator的GEMMQuantMode
class GEMMQuantMode(Enum):
    # 现有定义...

    # 新增Marlin加速内核
    awq_marlin  = QuantMapping(memory=0.5, compute=1.5, name="awq_marlin")  # AWQ + Marlin kernel
    gptq_marlin = QuantMapping(memory=0.5, compute=1.5, name="gptq_marlin") # GPTQ + Marlin kernel

    # 新增GGUF格式
    gguf        = QuantMapping(memory=0.5, compute=1, name="gguf")  # llama.cpp兼容格式

    # 新增bitsandbytes
    bitsandbytes = QuantMapping(memory=0.5, compute=1, name="bitsandbytes")  # 8-bit/4-bit

    # 新增auto-round
    auto_round  = QuantMapping(memory=0.5, compute=1, name="auto_round")  # Intel Auto-Round

    # 新增compressed-tensors
    compressed_tensors = QuantMapping(memory=0.5, compute=1, name="compressed_tensors")  # Ktransformers

    # 新增QoQ和w4afp8
    qoq         = QuantMapping(memory=0.25, compute=1.5, name="qoq")  # Quantization-on-Quantization
    w4afp8      = QuantMapping(memory=0.5, compute=2, name="w4afp8")  # 4-bit weights + FP8 activations
```

**映射示例**:
```json
{
  "parameters": {
    "quantization": ["awq_marlin", "gptq_marlin", "fp8", "w8a8_fp8"],
    "dtype": ["half", "auto"]
  }
}
```

---

## 2. KV Cache Quantization Mode

**概念**: KV Cache量化控制key-value缓存的存储精度，对内存占用影响巨大。

### AIConfigurator定义

```python
class KVCacheQuantMode(Enum):
    float16 = QuantMapping(memory=2, compute=0, name="float16")  # FP16 KV cache
    int8    = QuantMapping(memory=1, compute=0, name="int8")     # INT8 KV cache
    fp8     = QuantMapping(memory=1, compute=0, name="fp8")      # FP8 KV cache
```

**注意**: `compute=0` 表示KV cache量化不影响计算速度（仅影响内存带宽）

### 映射到SGLang参数

| AIConfigurator | SGLang --kv-cache-dtype | 内存节省 | CUDA要求 |
|----------------|------------------------|---------|----------|
| `float16` | `auto` 或 `bfloat16` | 0% (基准) | 所有GPU |
| `int8` | (不直接支持) | - | - |
| `fp8` | `fp8_e5m2` 或 `fp8_e4m3` | ~50% | CUDA 11.8+ |

### 扩展建议：SGLang支持但AIConfigurator未定义的KV Cache量化方法

```python
# 建议扩展到AIConfigurator的KVCacheQuantMode
class KVCacheQuantMode(Enum):
    # 现有定义...

    # 新增FP8变体
    fp8_e5m2 = QuantMapping(memory=1, compute=0, name="fp8_e5m2")  # FP8 E5M2格式
    fp8_e4m3 = QuantMapping(memory=1, compute=0, name="fp8_e4m3")  # FP8 E4M3格式

    # 新增FP4
    fp4_e2m1 = QuantMapping(memory=0.5, compute=0, name="fp4_e2m1")  # FP4 mxfp4 (CUDA 12.8+)

    # 新增bfloat16
    bfloat16 = QuantMapping(memory=2, compute=0, name="bfloat16")  # BF16 KV cache
```

**映射示例**:
```json
{
  "parameters": {
    "kv-cache-dtype": ["auto", "fp8_e5m2", "fp8_e4m3", "fp4_e2m1", "bf16"]
  }
}
```

**内存节省对比**:
- `fp8_e5m2` / `fp8_e4m3`: 相比FP16节省 ~50% 内存
- `fp4_e2m1`: 相比FP16节省 ~75% 内存
- 对长上下文场景（32K+ tokens）影响显著

---

## 3. FMHA Quantization Mode

**概念**: FMHA (Fused Multi-Head Attention) 量化控制attention计算的精度。

### AIConfigurator定义

```python
class FMHAQuantMode(Enum):
    float16   = QuantMapping(memory=0, compute=1, name="float16")    # FP16 attention
    fp8       = QuantMapping(memory=0, compute=2, name="fp8")        # FP8 attention (2x加速)
    fp8_block = QuantMapping(memory=1, compute=2, name="fp8_block")  # SGLang specific
```

**注意**: `memory=0` 表示FMHA量化主要影响计算速度，内存影响较小

### 映射到SGLang参数

| AIConfigurator | 对应SGLang行为 | 说明 |
|----------------|---------------|------|
| `float16` | `--dtype float16` | FP16 attention计算 |
| `fp8` | `--quantization fp8` | FP8量化（包含attention） |
| `fp8_block` | `--quantization modelopt_fp8` | ModelOpt FP8量化 |

**重要说明**: SGLang没有单独的"attention量化"参数，FMHA精度由以下决定：
1. **全局dtype**: `--dtype` 控制所有计算精度
2. **全局quantization**: `--quantization fp8` 会同时量化GEMM和FMHA
3. **Attention backend**: `--attention-backend` 可选择不同实现（flashinfer, fa3等）

### 映射建议

对于inference-autotuner，建议：
1. **不单独调参FMHA量化** - 跟随GEMM量化设置
2. **调参attention backend** - 可影响性能

```json
{
  "parameters": {
    "quantization": ["fp8", "modelopt_fp8"],
    "attention-backend": ["flashinfer", "fa3", "triton"]
  }
}
```

---

## 4. MoE Quantization Mode

**概念**: MoE (Mixture of Experts) 量化控制专家网络层的精度，仅对MoE模型有效。

### AIConfigurator定义

```python
class MoEQuantMode(Enum):
    float16       = QuantMapping(memory=2, compute=1, name="float16")    # w16a16
    fp8           = QuantMapping(memory=1, compute=2, name="fp8")        # w8a8 FP8
    int4_wo       = QuantMapping(memory=0.5, compute=1, name="int4_wo")  # w4a16
    fp8_block     = QuantMapping(memory=1, compute=2, name="fp8_block")  # TRT-LLM FP8
    w4afp8        = QuantMapping(memory=0.5, compute=2, name="w4afp8")   # w4a8 FP8
    nvfp4         = QuantMapping(memory=0.5, compute=4, name="nvfp4")    # NVIDIA FP4
    w4a16_mxfp4   = QuantMapping(memory=0.5, compute=1, name="w4a16_mxfp4")  # mxfp4 native
```

### 映射到SGLang参数

| AIConfigurator | SGLang参数 | 说明 |
|----------------|-----------|------|
| `float16` | `--dtype float16` | 全精度MoE |
| `fp8` | `--quantization fp8` | FP8量化MoE |
| `int4_wo` | `--quantization awq` 或 `gptq` | 4-bit MoE权重 |
| `fp8_block` | `--quantization modelopt_fp8` | ModelOpt FP8 |
| `w4afp8` | `--quantization w4afp8` | 4-bit weights + FP8 activations |
| `nvfp4` | `--quantization modelopt_fp4` | FP4量化 (Blackwell) |
| `w4a16_mxfp4` | `--quantization mxfp4` | Microscaling FP4 |

**重要说明**: SGLang通过以下参数优化MoE：
- `--moe-expert-tp-size`: MoE expert tensor parallelism
- `--moe-ep-size`: MoE expert parallelism size
- Backend选择: `--moe-runner-backend` (如 `flashinfer_mxfp4`)

### MoE特定调参建议

```json
{
  "parameters": {
    "quantization": ["fp8", "w4afp8", "mxfp4"],
    "moe-runner-backend": ["auto", "flashinfer_cutlass", "flashinfer_mxfp4"]
  }
}
```

---

## 综合映射总结

### 参数组合建议

#### 配置1: 高精度低内存（适合开发测试）
```json
{
  "gemm_quant_mode": "float16",
  "kvcache_quant_mode": "fp8",
  "fmha_quant_mode": "float16",
  "moe_quant_mode": "float16"
}
```
**SGLang等效配置**:
```bash
--dtype bfloat16 --kv-cache-dtype fp8_e5m2
```

#### 配置2: 平衡性能（推荐生产环境）
```json
{
  "gemm_quant_mode": "fp8",
  "kvcache_quant_mode": "fp8",
  "fmha_quant_mode": "fp8",
  "moe_quant_mode": "fp8"
}
```
**SGLang等效配置**:
```bash
--quantization fp8 --kv-cache-dtype fp8_e5m2
```

#### 配置3: 极致压缩（内存受限场景）
```json
{
  "gemm_quant_mode": "int4_wo",
  "kvcache_quant_mode": "fp8",
  "fmha_quant_mode": "float16",
  "moe_quant_mode": "int4_wo"
}
```
**SGLang等效配置**:
```bash
--model-path <pre-quantized-awq-model> --kv-cache-dtype fp8_e5m2 --dtype half
```

#### 配置4: Blackwell GPU专属（最高性能）
```json
{
  "gemm_quant_mode": "nvfp4",
  "kvcache_quant_mode": "fp8",
  "fmha_quant_mode": "fp8",
  "moe_quant_mode": "nvfp4"
}
```
**SGLang等效配置**:
```bash
--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
```

---

## Inference-Autotuner任务配置示例

### 示例1: 调优KV Cache量化影响

```json
{
  "task_name": "kvcache-quant-comparison",
  "model": {
    "id_or_path": "meta-llama/Llama-3.2-1B-Instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "parameters": {
    "tp-size": [1, 2],
    "kv-cache-dtype": ["auto", "fp8_e5m2", "fp8_e4m3", "fp4_e2m1", "bf16"],
    "mem-fraction-static": [0.85, 0.9]
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "maximize_throughput",
    "max_iterations": 30
  },
  "benchmark": {
    "num_concurrency": [1, 4, 8]
  }
}
```

**对应aiconfigurator维度**: `kvcache_quant_mode`

### 示例2: 调优GEMM量化方法

```json
{
  "task_name": "gemm-quant-comparison",
  "model": {
    "id_or_path": "meta-llama/Llama-3.2-1B-Instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "parameters": {
    "quantization": ["fp8", "w8a8_fp8"],
    "dtype": ["auto", "bfloat16"],
    "tp-size": [1, 2, 4]
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "balanced",
    "max_iterations": 24
  }
}
```

**对应aiconfigurator维度**: `gemm_quant_mode`

### 示例3: MoE模型量化调优

```json
{
  "task_name": "moe-quant-optimization",
  "model": {
    "id_or_path": "deepseek-ai/DeepSeek-V3",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "parameters": {
    "quantization": ["fp8", "w4afp8", "mxfp4"],
    "kv-cache-dtype": ["fp8_e5m2"],
    "moe-runner-backend": ["auto", "flashinfer_cutlass", "flashinfer_mxfp4"],
    "tp-size": [8],
    "moe-tp-size": [1, 2]
  },
  "optimization": {
    "strategy": "bayesian",
    "objective": "maximize_throughput",
    "max_iterations": 50
  }
}
```

**对应aiconfigurator维度**: `moe_quant_mode` + `gemm_quant_mode` + `kvcache_quant_mode`

### 示例4: 完整量化配置空间探索

```json
{
  "task_name": "full-quant-space-exploration",
  "model": {
    "id_or_path": "meta-llama/Llama-3.2-1B-Instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "parameters": {
    "dtype": ["bfloat16", "float16"],
    "quantization": ["fp8", "w8a8_fp8", "modelopt_fp8"],
    "kv-cache-dtype": ["auto", "fp8_e5m2", "bf16"],
    "tp-size": [1, 2, 4],
    "mem-fraction-static": [0.85, 0.9]
  },
  "optimization": {
    "strategy": "bayesian",
    "objective": "balanced",
    "max_iterations": 100
  },
  "slo": {
    "ttft": {
      "threshold": 1.0,
      "weight": 2.0,
      "hard_fail": false
    },
    "tpot": {
      "threshold": 0.05,
      "weight": 2.0,
      "hard_fail": false
    }
  }
}
```

**对应aiconfigurator维度**: 覆盖 `gemm_quant_mode` + `kvcache_quant_mode` + `fmha_quant_mode`

---

## 扩展参数集合建议

### 推荐添加到inference-autotuner的SGLang特定参数

```python
# 建议在 inference-autotuner 中支持的SGLang参数
SGLANG_QUANTIZATION_PARAMETERS = {
    # 基础量化参数
    "dtype": ["auto", "half", "float16", "bfloat16", "float", "float32"],

    # 主要量化方法
    "quantization": [
        "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin",
        "bitsandbytes", "gguf", "modelopt", "modelopt_fp8", "modelopt_fp4",
        "w8a8_int8", "w8a8_fp8", "w4afp8", "mxfp4", "auto-round"
    ],

    # KV Cache量化
    "kv-cache-dtype": ["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],

    # ModelOpt特定参数
    "modelopt-quant": ["fp8", "int4_awq", "w4a8_awq", "nvfp4", "nvfp4_awq"],

    # MoE优化参数
    "moe-runner-backend": [
        "auto", "deep_gemm", "triton", "flashinfer_trtllm",
        "flashinfer_cutlass", "flashinfer_mxfp4"
    ],

    # Attention backend
    "attention-backend": [
        "flashinfer", "fa3", "fa4", "triton", "torch_native"
    ],

    # Torchao量化（可选）
    "torchao-config": [
        "int8dq", "int8wo", "fp8wo", "int4wo-128", "int4wo-256"
    ]
}
```

### 参数优先级建议

对于有限的实验预算，建议按以下优先级调参：

**Tier 1 (最高影响)**:
1. `kv-cache-dtype` - 对内存和长上下文性能影响最大
2. `quantization` - 对整体性能和内存的平衡影响
3. `tp-size` - 并行策略的基础

**Tier 2 (中等影响)**:
4. `dtype` - 与quantization配合使用
5. `mem-fraction-static` - 内存分配策略
6. `moe-runner-backend` - 仅对MoE模型重要

**Tier 3 (特定场景)**:
7. `attention-backend` - 针对特定GPU架构优化
8. `modelopt-quant` - 需要NVIDIA ModelOpt依赖
9. `torchao-config` - 实验性功能

---

## 兼容性注意事项

### GPU架构限制

| 量化方法 | 最低GPU | CUDA版本 | PyTorch版本 |
|---------|--------|----------|-------------|
| FP8 (大多数) | Ampere | CUDA 11.8+ | 任意 |
| FP4 (mxfp4) | Hopper | CUDA 12.8+ | PyTorch 2.8+ |
| ModelOpt FP8 | Hopper | CUDA 11.8+ | 任意 |
| ModelOpt FP4 | Blackwell | CUDA 12.0+ | 任意 |
| AWQ/GPTQ | Pascal | CUDA 11.4+ | 任意 |

### 模型特定考虑

**DeepSeek V3/R1**:
- 模型已预量化为FP8
- 不要添加额外的 `--quantization` 参数
- 推荐配置: `--kv-cache-dtype fp8_e5m2`

**MoE模型**:
- 量化支持有限，可能需要跳过某些层
- 推荐使用 `mxfp4` 或 `w4afp8` 方法
- 避免量化 `mlp.gate` 层

**VLM（视觉-语言模型）**:
- 量化支持受限
- AWQ/auto_awq格式最可靠
- GPTQ可能有兼容性问题

---

## 总结

### 核心映射关系

| AIConfigurator维度 | SGLang主参数 | SGLang辅助参数 |
|-------------------|-------------|---------------|
| `gemm_quant_mode` | `--quantization` | `--dtype` |
| `kvcache_quant_mode` | `--kv-cache-dtype` | - |
| `fmha_quant_mode` | 跟随 `--quantization` | `--attention-backend` |
| `moe_quant_mode` | `--quantization` | `--moe-runner-backend` |

### 关键差异

1. **粒度差异**:
   - AIConfigurator: 细粒度控制（GEMM、FMHA、KV、MoE分离）
   - SGLang: 粗粒度控制（全局quantization + KV Cache分离）

2. **策略差异**:
   - AIConfigurator: 基于性能数据库的Cost Model
   - SGLang: 基于实际运行的Benchmark驱动

3. **应用场景**:
   - AIConfigurator: 离线预测和容量规划
   - SGLang + inference-autotuner: 在线实测和参数优化

### 集成建议

对于将两者集成的场景：

1. **使用inference-autotuner进行实测**: 获取真实硬件上的性能数据
2. **映射到aiconfigurator术语**: 将SGLang参数组合映射为4维量化配置
3. **填充aiconfigurator数据库**: 用实测数据扩展或校准Cost Model
4. **闭环优化**: 使用aiconfigurator预测 → inference-autotuner验证 → 更新数据库

这样可以结合两者优势：AIConfigurator的快速预测能力 + inference-autotuner的精确实测能力。
