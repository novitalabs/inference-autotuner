# Quantization Parameter Comparison: vLLM vs TensorRT-LLM vs SGLang

本文档对比三大主流LLM推理引擎的量化参数支持情况，为inference-autotuner提供跨引擎参数映射参考。

## 执行摘要

| 引擎 | 量化方法数量 | KV Cache选项 | 特色功能 | 最佳应用场景 |
|------|------------|-------------|---------|------------|
| **vLLM** | 34种 | 6种 | 最广泛的量化方法支持 | 研究、实验、多样化量化方法 |
| **TensorRT-LLM** | 18种 (QuantAlgo) | 3种 | 企业级优化、最佳性能 | 生产部署、NVIDIA GPU优化 |
| **SGLang** | 21种 | 6种 | Marlin加速内核、易用性 | 平衡性能和易用性 |

---

## 一、量化方法对比

### 1.1 vLLM量化方法 (34种)

**来源**: `/root/work/vllm/vllm/model_executor/layers/quantization/__init__.py:8-38`

```python
QuantizationMethods = Literal[
    "awq",                    # AWQ 4-bit量化
    "deepspeedfp",            # DeepSpeed FP量化
    "tpu_int8",               # TPU INT8量化
    "fp8",                    # FP8动态量化
    "ptpc_fp8",               # PTPC FP8
    "fbgemm_fp8",             # Facebook FBGEMM FP8
    "fp_quant",               # 浮点量化
    "modelopt",               # NVIDIA ModelOpt FP8
    "modelopt_fp4",           # NVIDIA ModelOpt FP4
    "bitblas",                # BitBLAS量化
    "gguf",                   # llama.cpp GGUF格式
    "gptq_marlin_24",         # GPTQ + Marlin + 2:4稀疏
    "gptq_marlin",            # GPTQ + Marlin加速
    "gptq_bitblas",           # GPTQ + BitBLAS
    "awq_marlin",             # AWQ + Marlin加速
    "gptq",                   # GPTQ量化
    "compressed-tensors",     # 压缩张量格式
    "bitsandbytes",           # bitsandbytes 8/4-bit
    "hqq",                    # Half-Quadratic Quantization
    "experts_int8",           # 专家层INT8量化
    "ipex",                   # Intel Extension for PyTorch
    "quark",                  # Qualcomm Quark量化
    "moe_wna16",              # MoE权重INT量化激活FP16
    "torchao",                # PyTorch AO量化
    "auto-round",             # Intel Auto-Round
    "rtn",                    # Round-To-Nearest
    "inc",                    # Intel Neural Compressor
    "mxfp4",                  # Microscaling FP4
    "petit_nvfp4",            # Petit NVIDIA FP4
]
```

**分类统计**:
- **Weight-only量化**: AWQ, GPTQ, GGUF, bitsandbytes, auto-round, rtn (6种)
- **Weight+Activation量化**: FP8系列 (7种), INT8系列 (3种)
- **优化内核**: Marlin系列 (3种), BitBLAS (2种)
- **硬件特定**: TPU INT8, IPEX (Intel), Quark (Qualcomm), ModelOpt (NVIDIA)
- **MoE专用**: experts_int8, moe_wna16
- **框架集成**: DeepSpeedFP, torchao, compressed-tensors

### 1.2 TensorRT-LLM量化方法 (18种QuantAlgo)

**来源**: `/root/work/TensorRT-LLM/tensorrt_llm/quantization/mode.py:23-43`

```python
class QuantAlgo(StrEnum):
    W8A16                                = auto()  # 8-bit权重，16-bit激活
    W4A16                                = auto()  # 4-bit权重，16-bit激活
    W4A16_AWQ                            = auto()  # AWQ 4-bit权重
    W4A8_AWQ                             = auto()  # AWQ 4-bit权重 + 8-bit激活
    W8A16_GPTQ                           = auto()  # GPTQ 8-bit权重
    W4A16_GPTQ                           = auto()  # GPTQ 4-bit权重
    W8A8_SQ_PER_CHANNEL                  = auto()  # SmoothQuant per-channel
    W8A8_SQ_PER_TENSOR_PLUGIN            = auto()  # SmoothQuant per-tensor
    W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN = auto()  # SmoothQuant per-channel+token
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN= auto()  # SmoothQuant per-channel+tensor
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN  = auto()  # SmoothQuant per-tensor+token
    W4A8_QSERVE_PER_GROUP                = auto()  # QServe per-group
    W4A8_QSERVE_PER_CHANNEL              = auto()  # QServe per-channel
    FP8                                  = auto()  # FP8 QDQ
    FP8_PER_CHANNEL_PER_TOKEN            = auto()  # FP8 rowwise
    FP8_BLOCK_SCALES                     = auto()  # FP8 block scales (DeepSeek)
    INT8                                 = auto()  # INT8量化
    NVFP4                                = auto()  # NVIDIA FP4 (Blackwell)
    MIXED_PRECISION                      = auto()  # 混合精度
    NO_QUANT                             = auto()  # 不量化
]
```

**分类统计**:
- **Weight-only**: W8A16, W4A16, W4A16_AWQ, W4A16_GPTQ, W8A16_GPTQ (5种)
- **SmoothQuant变体**: 5种精细粒度控制
- **FP8系列**: FP8, FP8_PER_CHANNEL_PER_TOKEN, FP8_BLOCK_SCALES (3种)
- **先进方法**: W4A8_AWQ, W4A8_QSERVE (2种), NVFP4
- **特殊**: MIXED_PRECISION, INT8

### 1.3 SGLang量化方法 (21种)

**来源**: `/root/work/sglang/python/sglang/srt/server_args.py:85-106`

```python
QUANTIZATION_CHOICES = [
    "awq",                # AWQ 4-bit
    "fp8",                # FP8量化
    "gptq",               # GPTQ量化
    "marlin",             # Marlin优化内核
    "gptq_marlin",        # GPTQ + Marlin
    "awq_marlin",         # AWQ + Marlin
    "bitsandbytes",       # bitsandbytes 8/4-bit
    "gguf",               # GGUF格式
    "modelopt",           # NVIDIA ModelOpt
    "modelopt_fp8",       # ModelOpt FP8
    "modelopt_fp4",       # ModelOpt FP4
    "petit_nvfp4",        # Petit NVIDIA FP4
    "w8a8_int8",          # 8-bit权重+激活 INT8
    "w8a8_fp8",           # 8-bit权重+激活 FP8
    "moe_wna16",          # MoE权重量化激活16-bit
    "qoq",                # Quantization-on-Quantization
    "w4afp8",             # 4-bit权重+FP8激活
    "mxfp4",              # Microscaling FP4
    "auto-round",         # Intel Auto-Round
    "compressed-tensors", # Ktransformers压缩
]
```

**分类统计**:
- **Weight-only**: AWQ, GPTQ, GGUF, bitsandbytes, auto-round (5种)
- **Weight+Activation**: fp8, w8a8_int8, w8a8_fp8, w4afp8 (4种)
- **Marlin加速**: marlin, gptq_marlin, awq_marlin (3种)
- **ModelOpt系列**: modelopt, modelopt_fp8, modelopt_fp4 (3种)
- **先进方法**: qoq, mxfp4, petit_nvfp4
- **MoE专用**: moe_wna16

---

## 二、KV Cache量化对比

### 2.1 vLLM KV Cache选项 (6种)

**来源**: `/root/work/vllm/vllm/config/cache.py:24`

```python
CacheDType = Literal["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]
```

| 选项 | 精度 | 内存占用 | 支持平台 | 说明 |
|------|------|---------|---------|------|
| `auto` | 跟随模型 | 100% (基准) | 全平台 | 默认使用模型dtype |
| `bfloat16` | BF16 | 100% | 全平台 | 显式BF16 KV cache |
| `fp8` | FP8 E4M3 | ~50% | CUDA 11.8+ | 默认FP8格式 |
| `fp8_e4m3` | FP8 E4M3 | ~50% | CUDA 11.8+ / ROCm | 4位指数3位尾数 |
| `fp8_e5m2` | FP8 E5M2 | ~50% | CUDA 11.8+ | 5位指数2位尾数 |
| `fp8_inc` | FP8 (INC) | ~50% | Intel Gaudi (HPU) | Intel Neural Compressor FP8 |

### 2.2 TensorRT-LLM KV Cache选项 (3种)

**来源**: `/root/work/TensorRT-LLM/tensorrt_llm/quantization/mode.py:47`

```python
KV_CACHE_QUANT_ALGO_LIST = [QuantAlgo.FP8, QuantAlgo.INT8, QuantAlgo.NVFP4]
```

| 选项 | 精度 | 内存占用 | 支持平台 | 说明 |
|------|------|---------|---------|------|
| (无) | FP16/BF16 | 100% (基准) | 全平台 | 默认不量化 |
| `INT8` | INT8 | ~50% | 全平台 | 整数量化 |
| `FP8` | FP8 | ~50% | CUDA 11.8+ | 浮点量化 |
| `NVFP4` | FP4 | ~25% | Blackwell | NVIDIA FP4量化 |

**QuantMode标志位**:
```python
class QuantMode(IntFlag):
    INT8_KV_CACHE    = auto()  # INT8 KV cache
    FP8_KV_CACHE     = auto()  # FP8 KV cache
    NVFP4_KV_CACHE   = auto()  # FP4 KV cache (Blackwell)
```

### 2.3 SGLang KV Cache选项 (6种)

**来源**: `/root/work/sglang/python/sglang/srt/server_args.py:2031`

```python
--kv-cache-dtype choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"]
```

| 选项 | 精度 | 内存占用 | 支持平台 | 说明 |
|------|------|---------|---------|------|
| `auto` | 跟随模型 | 100% (基准) | 全平台 | 默认使用模型dtype |
| `bf16` / `bfloat16` | BF16 | 100% | 全平台 | BF16 KV cache |
| `fp8_e5m2` | FP8 E5M2 | ~50% | CUDA 11.8+ | 推荐FP8格式 |
| `fp8_e4m3` | FP8 E4M3 | ~50% | CUDA 11.8+ | 备选FP8格式 |
| `fp4_e2m1` | FP4 (mxfp4) | ~25% | CUDA 12.8+, PyTorch 2.8+ | 极致压缩 |

---

## 三、量化方法详细对比

### 3.1 Weight-Only量化

#### AWQ (Activation-aware Weight Quantization)

| 引擎 | 方法名 | 精度 | 分组支持 | 加速内核 | 性能评级 |
|------|--------|------|---------|---------|---------|
| vLLM | `awq`, `awq_marlin` | 4-bit | ✅ | Marlin | ⭐⭐⭐⭐⭐ |
| TensorRT-LLM | `W4A16_AWQ`, `W4A8_AWQ` | 4-bit | ✅ | TensorRT | ⭐⭐⭐⭐⭐ |
| SGLang | `awq`, `awq_marlin` | 4-bit | ✅ | Marlin | ⭐⭐⭐⭐⭐ |

**特点**:
- 基于激活分布的权重量化
- Per-group量化，保留重要权重精度
- Marlin内核可提供1.5-2x加速
- 推荐用于4-bit量化场景

#### GPTQ (GPT Quantization)

| 引擎 | 方法名 | 精度 | 分组支持 | 加速内核 | 性能评级 |
|------|--------|------|---------|---------|---------|
| vLLM | `gptq`, `gptq_marlin`, `gptq_bitblas`, `gptq_marlin_24` | 2/3/4/8-bit | ✅ | Marlin, BitBLAS | ⭐⭐⭐⭐ |
| TensorRT-LLM | `W4A16_GPTQ`, `W8A16_GPTQ` | 4/8-bit | ✅ | TensorRT | ⭐⭐⭐⭐ |
| SGLang | `gptq`, `gptq_marlin` | 4/8-bit | ✅ | Marlin | ⭐⭐⭐⭐ |

**特点**:
- 基于Hessian矩阵的最优量化
- 支持多种bit-width (2/3/4/8)
- vLLM独有2:4稀疏支持 (`gptq_marlin_24`)
- 校准时间较长但精度好

#### GGUF (llama.cpp格式)

| 引擎 | 方法名 | 精度 | 跨平台 | 性能评级 |
|------|--------|------|--------|---------|
| vLLM | `gguf` | Q4_K_M, Q5_K_M, Q6_K, Q8_0等 | ✅ | ⭐⭐⭐ |
| TensorRT-LLM | (不直接支持) | - | - | - |
| SGLang | `gguf` | 多种量化级别 | ✅ | ⭐⭐⭐ |

**特点**:
- llama.cpp生态兼容
- 多种量化级别可选
- 适合边缘设备和跨平台部署
- TensorRT-LLM不直接支持（需转换）

### 3.2 Weight+Activation量化

#### FP8量化

| 引擎 | 方法名 | 变体数量 | 硬件要求 | 性能评级 |
|------|--------|---------|---------|---------|
| vLLM | `fp8`, `ptpc_fp8`, `fbgemm_fp8` | 3种 | Ampere+ (CUDA 11.8+) | ⭐⭐⭐⭐⭐ |
| TensorRT-LLM | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES` | 3种 | Ampere+ | ⭐⭐⭐⭐⭐ |
| SGLang | `fp8`, `w8a8_fp8` | 2种 | Ampere+ | ⭐⭐⭐⭐⭐ |

**TensorRT-LLM FP8变体详解**:
- `FP8`: 标准FP8 QDQ (量化-反量化)
- `FP8_PER_CHANNEL_PER_TOKEN`: Rowwise FP8，per-channel权重 + per-token激活
- `FP8_BLOCK_SCALES`: Block级缩放，DeepSeek V3专用 (1x128, 128x128块)

**vLLM FP8变体详解**:
- `fp8`: 通用动态FP8量化
- `ptpc_fp8`: Per-Tensor-Per-Channel FP8
- `fbgemm_fp8`: Facebook FBGEMM优化FP8

**特点**:
- Hopper GPU (H100) 可获得2x计算加速
- 内存节省50%
- 精度损失极小（相比FP16）
- 推荐用于大规模生产部署

#### INT8量化

| 引擎 | 方法名 | 变体数量 | SmoothQuant | 性能评级 |
|------|--------|---------|------------|---------|
| vLLM | `experts_int8` | 1种 (MoE专用) | ❌ | ⭐⭐⭐ |
| TensorRT-LLM | `W8A8_SQ_*` | 5种SQ变体 | ✅ | ⭐⭐⭐⭐ |
| SGLang | `w8a8_int8` | 1种 | ❌ | ⭐⭐⭐ |

**TensorRT-LLM SmoothQuant变体**:
1. `W8A8_SQ_PER_CHANNEL`: Per-channel权重量化
2. `W8A8_SQ_PER_TENSOR_PLUGIN`: Per-tensor量化（插件）
3. `W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN`: Per-channel权重 + per-token激活
4. `W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN`: Per-channel权重 + per-tensor激活
5. `W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN`: Per-tensor权重 + per-token激活

**特点**:
- TensorRT-LLM在INT8上有最丰富的粒度控制
- SmoothQuant通过平滑操作减少量化误差
- vLLM/SGLang的INT8支持相对有限

### 3.3 先进混合精度方法

#### W4A8 (4-bit权重 + 8-bit激活)

| 引擎 | 方法名 | 实现方式 | 性能评级 |
|------|--------|---------|---------|
| vLLM | (无直接支持) | - | - |
| TensorRT-LLM | `W4A8_AWQ`, `W4A8_QSERVE_*` | AWQ / QServe | ⭐⭐⭐⭐⭐ |
| SGLang | `w4afp8` | FP8激活 | ⭐⭐⭐⭐⭐ |

**特点**:
- TensorRT-LLM的QServe实现（`W4A8_QSERVE_PER_GROUP`, `W4A8_QSERVE_PER_CHANNEL`）
- 平衡内存占用和计算效率
- 适合大模型推理（70B+）

#### NVFP4 (NVIDIA FP4)

| 引擎 | 方法名 | 硬件要求 | 加速比 | 性能评级 |
|------|--------|---------|--------|---------|
| vLLM | `modelopt_fp4`, `mxfp4`, `petit_nvfp4` | Blackwell (B100/B200) | ~4x | ⭐⭐⭐⭐⭐ |
| TensorRT-LLM | `NVFP4` | Blackwell | ~4x | ⭐⭐⭐⭐⭐ |
| SGLang | `modelopt_fp4`, `mxfp4`, `petit_nvfp4` | Blackwell | ~4x | ⭐⭐⭐⭐⭐ |

**特点**:
- Blackwell架构专属（B100/B200 GPU）
- 理论4x计算加速 + 75%内存节省
- 三种实现：ModelOpt FP4, Microscaling FP4, Petit FP4
- 目前最前沿的量化技术

---

## 四、独家功能对比

### 4.1 vLLM独有功能

#### 1. **Hardware-Specific量化**

```python
"tpu_int8"      # Google TPU专用INT8
"ipex"          # Intel Extension for PyTorch (CPU/XPU)
"quark"         # Qualcomm Quark量化（移动端）
```

#### 2. **BitBLAS内核优化**

```python
"bitblas"       # BitBLAS通用量化
"gptq_bitblas"  # GPTQ + BitBLAS加速
```

- BitBLAS是微软开发的高性能量化内核
- 在某些场景下比Marlin更快

#### 3. **DeepSpeed集成**

```python
"deepspeedfp"   # DeepSpeed FP量化
```

- 与DeepSpeed训练框架深度集成
- 支持大规模分布式推理

#### 4. **GPTQ Marlin 2:4稀疏**

```python
"gptq_marlin_24"  # GPTQ + Marlin + 2:4结构化稀疏
```

- 结合量化和稀疏性
- 理论2x额外加速

### 4.2 TensorRT-LLM独有功能

#### 1. **精细粒度SmoothQuant控制**

5种SmoothQuant变体提供最细致的INT8量化控制：
- Per-channel vs Per-tensor权重
- Per-token vs Static激活
- 完全兼容TensorRT优化

#### 2. **Mixed Precision支持**

```python
QuantAlgo.MIXED_PRECISION  # 混合精度量化
```

- 不同层使用不同精度
- 关键层保持高精度，非关键层激进量化

#### 3. **FP8 Block Scales (DeepSeek专用)**

```python
QuantAlgo.FP8_BLOCK_SCALES  # 1x128_128x128 block scales
```

- 针对DeepSeek V3架构优化
- Block级缩放因子，降低量化误差

### 4.3 SGLang独有功能

#### 1. **Quantization-on-Quantization (QoQ)**

```python
"qoq"  # 二次量化，进一步压缩
```

- 对已量化模型再次量化
- 适合极端内存受限场景

#### 2. **Ktransformers压缩张量**

```python
"compressed-tensors"  # Ktransformers专用格式
```

- 与Ktransformers框架集成
- 支持自定义压缩策略

---

## 五、性能与兼容性矩阵

### 5.1 GPU架构兼容性

| 量化方法 | Pascal (GTX 10xx) | Volta (V100) | Ampere (A100) | Hopper (H100) | Blackwell (B100/200) |
|---------|-------------------|--------------|---------------|---------------|---------------------|
| AWQ/GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 | ❌ | ❌ | ✅ | ✅ (2x加速) | ✅ (2x加速) |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| NVFP4 | ❌ | ❌ | ❌ | ❌ | ✅ (4x加速) |
| Marlin | ✅ | ✅ | ✅ | ✅ | ✅ |

### 5.2 CUDA/PyTorch版本要求

| 量化方法 | 最低CUDA版本 | 推荐CUDA版本 | PyTorch版本 |
|---------|-------------|-------------|------------|
| FP8 (大多数) | CUDA 11.8 | CUDA 12.0+ | 任意 |
| FP8_inc (Intel) | - | - | 任意 |
| FP4 (mxfp4) | CUDA 12.8 | CUDA 12.8+ | PyTorch 2.8+ |
| NVFP4 | CUDA 12.0 | CUDA 12.4+ | 任意 |
| AWQ/GPTQ | CUDA 11.4 | CUDA 12.0+ | 任意 |

### 5.3 模型类型兼容性

| 量化方法 | LLaMA/GPT | MoE (DeepSeek/Mixtral) | VLM (Qwen-VL) | 注意事项 |
|---------|-----------|----------------------|--------------|---------|
| AWQ | ✅ | ⚠️ | ✅ | MoE: 避免量化gate层 |
| GPTQ | ✅ | ⚠️ | ⚠️ | VLM: 可能精度损失 |
| FP8 | ✅ | ✅ | ✅ | 最佳通用选择 |
| NVFP4 | ✅ | ✅ | ⚠️ | VLM支持有限 |
| experts_int8 | ❌ | ✅ | ❌ | MoE专用 |

---

## 六、Inference-Autotuner调优建议

### 6.1 跨引擎参数映射

#### 场景1: 通用4-bit量化

```json
{
  "vLLM": {
    "quantization": "awq_marlin",
    "kv_cache_dtype": "fp8_e5m2"
  },
  "TensorRT-LLM": {
    "quant_algo": "W4A16_AWQ",
    "kv_cache_quant_algo": "FP8"
  },
  "SGLang": {
    "quantization": "awq_marlin",
    "kv-cache-dtype": "fp8_e5m2"
  }
}
```

#### 场景2: 高性能FP8量化

```json
{
  "vLLM": {
    "quantization": "fp8",
    "kv_cache_dtype": "fp8_e4m3"
  },
  "TensorRT-LLM": {
    "quant_algo": "FP8",
    "kv_cache_quant_algo": "FP8"
  },
  "SGLang": {
    "quantization": "fp8",
    "kv-cache-dtype": "fp8_e4m3"
  }
}
```

#### 场景3: 极致压缩 (Blackwell GPU)

```json
{
  "vLLM": {
    "quantization": "modelopt_fp4",
    "kv_cache_dtype": "fp8_e5m2"
  },
  "TensorRT-LLM": {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": "NVFP4"
  },
  "SGLang": {
    "quantization": "modelopt_fp4",
    "kv-cache-dtype": "fp4_e2m1"
  }
}
```

### 6.2 Autotuner任务配置示例

#### 示例1: 跨引擎量化对比

```json
{
  "task_name": "cross-engine-quantization-comparison",
  "variants": [
    {
      "engine": "vllm",
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "parameters": {
        "quantization": ["awq_marlin", "fp8", "gptq_marlin"],
        "kv-cache-dtype": ["auto", "fp8_e5m2"],
        "tensor-parallel-size": [1, 2]
      }
    },
    {
      "engine": "sglang",
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "parameters": {
        "quantization": ["awq_marlin", "fp8", "gptq_marlin"],
        "kv-cache-dtype": ["auto", "fp8_e5m2"],
        "tp-size": [1, 2]
      }
    },
    {
      "engine": "tensorrt-llm",
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "parameters": {
        "quant-algo": ["W4A16_AWQ", "FP8"],
        "kv-cache-quant-algo": ["FP8"],
        "tp-size": [1, 2]
      }
    }
  ],
  "optimization": {
    "strategy": "grid_search",
    "objective": "maximize_throughput",
    "max_iterations": 50
  }
}
```

#### 示例2: vLLM量化方法全覆盖

```json
{
  "task_name": "vllm-quantization-sweep",
  "engine": "vllm",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "parameters": {
    "quantization": [
      "awq", "awq_marlin",
      "gptq", "gptq_marlin", "gptq_bitblas",
      "fp8", "fbgemm_fp8",
      "bitsandbytes",
      "auto-round"
    ],
    "kv-cache-dtype": ["auto", "fp8_e5m2", "fp8_e4m3"],
    "tensor-parallel-size": [1, 2, 4]
  },
  "optimization": {
    "strategy": "bayesian",
    "objective": "balanced",
    "max_iterations": 100
  },
  "slo": {
    "ttft": {"threshold": 1.0, "weight": 2.0},
    "tpot": {"threshold": 0.05, "weight": 2.0}
  }
}
```

### 6.3 推荐调优优先级

**Tier 1 (最高ROI)**:
1. **KV Cache dtype** - 对内存和长上下文性能影响最大
2. **Weight量化方法** - AWQ vs GPTQ vs FP8
3. **Tensor Parallelism** - 跨引擎参数名不同但效果类似

**Tier 2 (中等ROI)**:
4. **加速内核** - Marlin vs BitBLAS vs native
5. **FP8变体** - Per-channel vs Per-tensor vs Block scales
6. **内存分配** - gpu_memory_utilization / mem-fraction-static

**Tier 3 (特定场景)**:
7. **MoE专用量化** - experts_int8, moe_wna16
8. **硬件特定优化** - NVFP4 (Blackwell), IPEX (Intel)
9. **框架集成** - DeepSpeedFP, compressed-tensors

---

## 七、总结与建议

### 7.1 引擎选择指南

| 使用场景 | 推荐引擎 | 理由 |
|---------|---------|------|
| **研究与实验** | vLLM | 34种量化方法，最大灵活性 |
| **生产部署 (NVIDIA)** | TensorRT-LLM | 最佳性能，企业级支持 |
| **快速原型** | SGLang | 易用性好，Marlin加速内核 |
| **跨硬件平台** | vLLM | TPU/Intel/Qualcomm支持 |
| **MoE模型** | TensorRT-LLM / SGLang | 专用MoE优化 |
| **极致性能 (H100)** | TensorRT-LLM | FP8精细控制 |

### 7.2 量化方法选择指南

| 目标 | 推荐方法 | 引擎支持 |
|------|---------|---------|
| **最佳精度** | FP8 | 全部 |
| **最小内存** | AWQ/GPTQ 4-bit | 全部 |
| **最快推理 (4-bit)** | AWQ + Marlin | vLLM, SGLang |
| **最快推理 (8-bit)** | FP8 | 全部 |
| **极致压缩 (Blackwell)** | NVFP4 | 全部 |
| **跨平台兼容** | GGUF | vLLM, SGLang |
| **MoE优化** | experts_int8, FP8 | vLLM, TensorRT-LLM |

### 7.3 Inference-Autotuner集成建议

1. **参数标准化**:
   - 定义统一的量化配置抽象层
   - 自动映射到各引擎的具体参数

2. **性能基准库**:
   - 为常见模型建立跨引擎性能数据库
   - 指导用户快速选择最优配置

3. **智能推荐**:
   - 根据硬件(GPU架构)自动过滤不支持的方法
   - 根据模型类型(LLM/MoE/VLM)推荐最佳量化策略

4. **结果可视化**:
   - 跨引擎性能对比图表
   - 量化方法的内存-精度-速度三维权衡分析

---

## 附录A：快速参考表

### A.1 量化方法速查

| 方法 | vLLM | TensorRT-LLM | SGLang | 内存占用 | 加速比 |
|------|------|--------------|--------|---------|--------|
| AWQ | ✅ | ✅ W4A16_AWQ | ✅ | ~25% | ~1.5x |
| AWQ + Marlin | ✅ | ✅ | ✅ | ~25% | ~2x |
| GPTQ | ✅ | ✅ | ✅ | ~25% | ~1.5x |
| GPTQ + Marlin | ✅ | ❌ | ✅ | ~25% | ~2x |
| FP8 | ✅ | ✅ | ✅ | ~50% | ~2x (H100) |
| INT8 SmoothQuant | ❌ | ✅ (5变体) | ✅ | ~50% | ~1.5x |
| NVFP4 | ✅ | ✅ | ✅ | ~12.5% | ~4x (B100) |
| GGUF | ✅ | ❌ | ✅ | ~25-50% | ~1.3x |

### A.2 KV Cache选项速查

| 选项 | vLLM | TensorRT-LLM | SGLang | 内存节省 | 精度损失 |
|------|------|--------------|--------|---------|---------|
| auto/默认 | ✅ | ✅ | ✅ | 0% | 0% |
| BF16 | ✅ | ❌ | ✅ | 0% | 0% |
| FP8 | ✅ (3变体) | ✅ | ✅ (2变体) | ~50% | 极小 |
| INT8 | ❌ | ✅ | ❌ | ~50% | 小 |
| FP4 | ❌ | ✅ NVFP4 | ✅ | ~75% | 中等 |

---

**文档版本**: 1.0
**更新日期**: 2025-01-XX
**维护**: inference-autotuner项目组
