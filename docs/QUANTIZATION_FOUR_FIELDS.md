# Four-Field Quantization Configuration Schema

**Purpose**: Define runtime quantization configuration with four orthogonal fields: GEMM, KV Cache, Attention, and MoE.

**Design Principles**:
1. **Four Orthogonal Dimensions**: Each field controls a specific computation component
2. **Runtime Only**: Excludes offline quantization (AWQ, GPTQ, GGUF auto-detected from model)
3. **Three-Engine Universal**: All parameters work across vLLM, TensorRT-LLM, SGLang
4. **Mutually Exclusive Options**: Single dtype choice per field

---

## Four Orthogonal Fields

### Field 1: GEMM Dtype

**Purpose**: Data type for General Matrix Multiplication (linear layers, projections).

**Parameter Name**: `gemm_dtype`

**Scope**: Controls computation precision for:
- Input projections (Q, K, V)
- Output projections
- MLP layers (gate, up, down)
- Embedding layers

**Valid Options**:

| Option | Description | Memory | Compute Speed | Quality | Hardware |
|--------|-------------|--------|---------------|---------|----------|
| `auto` | Follow model default (FP16/BF16) | Baseline | 1x | Perfect | Any |
| `float16` | Explicit FP16 | 100% | 1x | Perfect | Any |
| `bfloat16` | Explicit BF16 | 100% | 1x | Perfect | Any |
| `float32` | FP32 (high precision) | 200% | 0.5x | Perfect | Any |
| `fp8` | FP8 dynamic quantization (W8A8) | 50% | 1.5-2x | Good | Ampere+ |
| `int8` | INT8 quantization | 50% | 1.2-1.5x | Moderate | Any |

**Notes**:
- `fp8` option enables **dynamic W8A8** quantization (weights + activations)
- Only applies to **unquantized models** (ignored for AWQ/GPTQ/GGUF models)
- `auto` is recommended (lets engine choose based on model)

---

### Field 2: KV Cache Dtype

**Purpose**: Data type for storing key-value cache tensors.

**Parameter Name**: `kvcache_dtype`

**Scope**: Controls storage precision for:
- Key cache (K projections from all layers)
- Value cache (V projections from all layers)

**Valid Options**:

| Option | Description | Memory Savings | Quality | Hardware |
|--------|-------------|----------------|---------|----------|
| `auto` | Follow model default (FP16/BF16) | 0% | Perfect | Any |
| `fp16` | Explicit FP16 | 0% | Perfect | Any |
| `bfloat16` | Explicit BF16 | 0% | Perfect | Any |
| `fp8` | FP8 (auto-select E4M3/E5M2) | ~50% | Excellent | Ampere+ |
| `fp8_e5m2` | FP8 E5M2 (5-bit exponent) | ~50% | Excellent | Ampere+ |
| `fp8_e4m3` | FP8 E4M3 (4-bit exponent) | ~50% | Excellent | Ampere+ |
| `int8` | INT8 quantization | ~50% | Good | Any |
| `int4` | INT4 quantization | ~75% | Moderate | Any |

**Notes**:
- **Always runtime-configurable** (independent of model quantization)
- Can be used with any model (AWQ, GPTQ, unquantized, etc.)
- `fp8_e5m2` recommended for best quality
- `fp8_e4m3` recommended for maximum hardware compatibility

---

### Field 3: Attention Dtype

**Purpose**: Data type for attention mechanism computations.

**Parameter Name**: `attention_dtype`

**Scope**: Controls computation precision for:
- QK^T matrix multiplication
- Softmax computation
- Attention weights × V multiplication

**Valid Options**:

| Option | Description | Memory | Quality | Hardware | Engine Support |
|--------|-------------|--------|---------|----------|----------------|
| `auto` | Follow model default | Baseline | Perfect | Any | All |
| `float16` | FP16 attention | 100% | Perfect | Any | All |
| `bfloat16` | BF16 attention | 100% | Perfect | Any | All |
| `fp8` | FP8 attention (FMHA) | 50% | Excellent | Ampere+ | TRT-LLM, SGLang |
| `fp8_e5m2` | FP8 E5M2 attention | 50% | Excellent | Ampere+ | TRT-LLM, SGLang |
| `fp8_e4m3` | FP8 E4M3 attention | 50% | Excellent | Ampere+ | TRT-LLM, SGLang |
| `fp8_block` | FP8 block-wise (experimental) | 50% | Excellent | Ampere+ | TRT-LLM only |

**Notes**:
- Separate from GEMM dtype (attention can use different precision than linear layers)
- **Supported by TensorRT-LLM and SGLang** (vLLM falls back to GEMM dtype)
- TensorRT-LLM: Uses `--fmha-quant-algo` parameter
- SGLang: Requires FP8-capable attention backend (FlashInfer, FlashAttention 3)
- `fp8_block`: Block-wise quantization for better quality (TRT-LLM experimental feature)

---

### Field 4: MoE Dtype

**Purpose**: Data type for Mixture-of-Experts computation.

**Parameter Name**: `moe_dtype`

**Scope**: Controls computation precision for:
- Expert router (gating network)
- Expert layers (each expert's MLP)
- Expert aggregation

**Valid Options**:

| Option | Description | Memory | Quality | Hardware | Backend |
|--------|-------------|--------|---------|----------|---------|
| `auto` | Follow model default | Baseline | Perfect | Any | Any |
| `float16` | FP16 experts | 100% | Perfect | Any | Any |
| `bfloat16` | BF16 experts | 100% | Perfect | Any | Any |
| `fp8` | FP8 experts (W8A8) | 50% | Excellent | Ampere+ | FlashInfer |
| `w4afp8` | 4-bit weights + FP8 activations | 37.5% | Good | Ampere+ | FlashInfer |
| `mxfp4` | MXFP4 (Blackwell) | 25% | Good | Blackwell | FlashInfer |
| `int8` | INT8 experts | 50% | Good | Any | Cutlass |

**Notes**:
- **Only applicable to MoE models** (Mixtral, DeepSeek-V2/V3, etc.)
- Ignored for dense models
- **Only fully supported by SGLang** (vLLM has limited support, TensorRT-LLM uses GEMM dtype)
- Requires specific MoE backend (FlashInfer, Cutlass)

---

## Complete Configuration Schema

### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Four-Field Quantization Configuration",
  "type": "object",
  "properties": {
    "quantization_config": {
      "type": "object",
      "properties": {
        "gemm_dtype": {
          "type": "string",
          "enum": ["auto", "float16", "bfloat16", "float32", "fp8", "int8"],
          "description": "GEMM computation precision (linear layers)",
          "default": "auto"
        },
        "kvcache_dtype": {
          "type": "string",
          "enum": ["auto", "fp16", "bfloat16", "fp8", "fp8_e5m2", "fp8_e4m3", "int8", "int4"],
          "description": "KV cache storage precision",
          "default": "auto"
        },
        "attention_dtype": {
          "type": "string",
          "enum": ["auto", "float16", "bfloat16", "fp8", "fp8_e5m2", "fp8_e4m3", "fp8_block"],
          "description": "Attention mechanism computation precision",
          "default": "auto"
        },
        "moe_dtype": {
          "type": "string",
          "enum": ["auto", "float16", "bfloat16", "fp8", "w4afp8", "mxfp4", "int8"],
          "description": "MoE expert computation precision (MoE models only)",
          "default": "auto"
        }
      },
      "required": ["gemm_dtype", "kvcache_dtype", "attention_dtype", "moe_dtype"]
    }
  }
}
```

### TypeScript Interface

```typescript
interface FourFieldQuantizationConfig {
  quantization_config: {
    gemm_dtype: "auto" | "float16" | "bfloat16" | "float32" | "fp8" | "int8";
    kvcache_dtype: "auto" | "fp16" | "bfloat16" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | "int8" | "int4";
    attention_dtype: "auto" | "float16" | "bfloat16" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | "fp8_block";
    moe_dtype: "auto" | "float16" | "bfloat16" | "fp8" | "w4afp8" | "mxfp4" | "int8";
  };
}
```

---

## Engine-Specific Parameter Mapping

### vLLM Parameter Mapping

```python
def map_to_vllm_args(config: dict) -> dict:
    """Map four-field config to vLLM CLI arguments."""
    args = {}

    # GEMM dtype
    gemm_dtype = config["gemm_dtype"]
    if gemm_dtype == "fp8":
        args["--quantization"] = "fp8"  # W8A8 dynamic quantization
        args["--dtype"] = "auto"
    elif gemm_dtype in ["int8"]:
        args["--quantization"] = "int8"
        args["--dtype"] = "auto"
    else:
        args["--dtype"] = gemm_dtype

    # KV cache dtype
    args["--kv-cache-dtype"] = config["kvcache_dtype"]

    # Attention dtype (vLLM v1 only, experimental)
    if config["attention_dtype"] != "auto":
        # vLLM doesn't have explicit attention dtype control yet
        # Falls back to GEMM dtype
        pass

    # MoE dtype (vLLM has limited MoE support)
    # Falls back to GEMM dtype

    return args
```

**Example Mapping**:
```json
Input:  {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "auto", "moe_dtype": "auto"}
Output: {"--quantization": "fp8", "--dtype": "auto", "--kv-cache-dtype": "fp8_e5m2"}
```

**vLLM Limitations**:
- ❌ No explicit attention dtype control (uses GEMM dtype)
- ⚠️ Limited MoE dtype control (uses GEMM dtype)

---

### TensorRT-LLM Parameter Mapping

```python
def map_to_tensorrt_llm_args(config: dict) -> dict:
    """Map four-field config to TensorRT-LLM arguments."""
    args = {}

    # GEMM dtype
    gemm_dtype = config["gemm_dtype"]
    if gemm_dtype == "fp8":
        args["--quant-algo"] = "FP8"
    elif gemm_dtype == "int8":
        args["--quant-algo"] = "INT8"
    # Note: TensorRT-LLM auto-detects model dtype, no explicit --dtype flag

    # KV cache dtype
    kvcache_dtype = config["kvcache_dtype"]
    if "fp8" in kvcache_dtype:
        args["--kv-cache-quant-algo"] = "FP8"
    elif kvcache_dtype == "int8":
        args["--kv-cache-quant-algo"] = "INT8"
    elif kvcache_dtype == "int4":
        args["--kv-cache-quant-algo"] = "INT4"

    # Attention dtype (FMHA quantization)
    attention_dtype = config["attention_dtype"]
    if attention_dtype == "fp8" or "fp8" in attention_dtype:
        args["--fmha-quant-algo"] = "FP8"
    elif attention_dtype == "fp8_block":
        args["--fmha-quant-algo"] = "FP8_BLOCK"  # Block-wise FP8 (experimental)
    # Note: float16/bfloat16 don't need explicit FMHA flag (default behavior)

    # MoE dtype
    # TensorRT-LLM doesn't support separate MoE dtype, uses GEMM dtype

    return args
```

**Example Mapping**:
```json
Input:  {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "fp8", "moe_dtype": "fp8"}
Output: {
  "--quant-algo": "FP8",
  "--kv-cache-quant-algo": "FP8",
  "--fmha-quant-algo": "FP8"
}
```

**TensorRT-LLM Features**:
- ✅ **Supports separate FMHA (attention) quantization** via `--fmha-quant-algo`
- ❌ No separate MoE dtype (uses GEMM dtype)
- ℹ️ Auto-detects model dtype (no explicit --dtype flag)

---

### SGLang Parameter Mapping

```python
def map_to_sglang_args(config: dict) -> dict:
    """Map four-field config to SGLang CLI arguments."""
    args = {}

    # GEMM dtype
    gemm_dtype = config["gemm_dtype"]
    if gemm_dtype == "fp8":
        args["--quantization"] = "fp8"
        args["--dtype"] = "auto"
    elif gemm_dtype == "int8":
        args["--quantization"] = "int8"
        args["--dtype"] = "auto"
    else:
        args["--dtype"] = gemm_dtype

    # KV cache dtype
    args["--kv-cache-dtype"] = config["kvcache_dtype"]

    # Attention dtype (SGLang supports separate attention dtype)
    attention_dtype = config["attention_dtype"]
    if attention_dtype != "auto":
        # SGLang can use FP8 attention with FlashInfer backend
        if attention_dtype in ["fp8", "fp8_e5m2", "fp8_e4m3"]:
            args["--attention-backend"] = "flashinfer"
            # FP8 attention enabled automatically with FlashInfer + FP8 KV cache

    # MoE dtype (SGLang has full MoE support)
    moe_dtype = config["moe_dtype"]
    if moe_dtype == "fp8":
        # Enable FP8 MoE with FlashInfer backend
        args["--moe-runner-backend"] = "flashinfer_cutlass"
    elif moe_dtype == "w4afp8":
        args["--quantization"] = "w4afp8"  # Override GEMM quantization for MoE
        args["--moe-runner-backend"] = "flashinfer_mxfp4"
    elif moe_dtype == "mxfp4":
        args["--quantization"] = "mxfp4"
        args["--moe-runner-backend"] = "flashinfer_mxfp4"

    return args
```

**Example Mapping**:
```json
Input:  {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "fp8", "moe_dtype": "w4afp8"}
Output: {
  "--quantization": "w4afp8",
  "--dtype": "auto",
  "--kv-cache-dtype": "fp8_e5m2",
  "--attention-backend": "flashinfer",
  "--moe-runner-backend": "flashinfer_mxfp4"
}
```

**SGLang Advantages**:
- ✅ Supports separate attention dtype
- ✅ Full MoE dtype control
- ✅ Most flexible configuration

---

## Field-to-Parameter Correspondence Table

### vLLM

| Field | vLLM Parameter | Notes |
|-------|----------------|-------|
| `gemm_dtype: auto/float16/bfloat16/float32` | `--dtype <value>` | Direct mapping |
| `gemm_dtype: fp8` | `--quantization fp8 --dtype auto` | W8A8 dynamic quant |
| `gemm_dtype: int8` | `--quantization int8 --dtype auto` | INT8 quantization |
| `kvcache_dtype` | `--kv-cache-dtype <value>` | Direct mapping |
| `attention_dtype` | ❌ Not supported | Falls back to GEMM dtype |
| `moe_dtype` | ❌ Limited support | Falls back to GEMM dtype |

### TensorRT-LLM

| Field | TensorRT-LLM Parameter | Notes |
|-------|------------------------|-------|
| `gemm_dtype: fp8` | `--quant-algo FP8` | W8A8 quantization |
| `gemm_dtype: int8` | `--quant-algo INT8` | INT8 quantization |
| `gemm_dtype: auto/float16/bfloat16` | (auto-detected) | No explicit flag |
| `kvcache_dtype: fp8*` | `--kv-cache-quant-algo FP8` | Doesn't distinguish E4M3/E5M2 |
| `kvcache_dtype: int8` | `--kv-cache-quant-algo INT8` | INT8 KV cache |
| `kvcache_dtype: int4` | `--kv-cache-quant-algo INT4` | INT4 KV cache (unique to TRT) |
| `attention_dtype: fp8*` | `--fmha-quant-algo FP8` | ✅ **Supports FMHA quantization** |
| `attention_dtype: fp8_block` | `--fmha-quant-algo FP8_BLOCK` | Block-wise FP8 (experimental) |
| `attention_dtype: float16/bfloat16` | (default behavior) | No explicit flag needed |
| `moe_dtype` | ❌ Not supported | Uses GEMM dtype |

### SGLang

| Field | SGLang Parameter | Notes |
|-------|------------------|-------|
| `gemm_dtype: auto/float16/bfloat16` | `--dtype <value>` | Direct mapping |
| `gemm_dtype: fp8` | `--quantization fp8 --dtype auto` | W8A8 dynamic quant |
| `gemm_dtype: int8` | `--quantization int8 --dtype auto` | INT8 quantization |
| `kvcache_dtype` | `--kv-cache-dtype <value>` | Direct mapping |
| `attention_dtype: fp8*` | `--attention-backend flashinfer` | FP8 attention with FlashInfer |
| `moe_dtype: fp8` | `--moe-runner-backend flashinfer_cutlass` | FP8 MoE experts |
| `moe_dtype: w4afp8` | `--quantization w4afp8 --moe-runner-backend flashinfer_mxfp4` | 4-bit weights + FP8 activations |
| `moe_dtype: mxfp4` | `--quantization mxfp4 --moe-runner-backend flashinfer_mxfp4` | MXFP4 MoE (Blackwell) |

---

## Configuration Examples

### Example 1: Standard FP16 (No Quantization)

```json
{
  "gemm_dtype": "auto",
  "kvcache_dtype": "auto",
  "attention_dtype": "auto",
  "moe_dtype": "auto"
}
```

**Behavior**:
- All computations at model default precision (FP16/BF16)
- No runtime quantization
- Works with any model

**Performance**: Memory: 100%, Throughput: 1x, Quality: Perfect

---

### Example 2: KV Cache Only Quantization (Recommended)

```json
{
  "gemm_dtype": "auto",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "auto",
  "moe_dtype": "auto"
}
```

**Behavior**:
- GEMM: FP16/BF16 (no quantization)
- KV Cache: FP8 (50% memory savings)
- Attention: FP16/BF16 (uses GEMM dtype)
- MoE: FP16/BF16 (uses GEMM dtype)

**Performance**: Memory: ~75%, Throughput: 1x, Quality: Near-perfect

**Recommendation**: **Best balance** of memory savings and quality.

---

### Example 3: Full FP8 (Maximum Throughput)

```json
{
  "gemm_dtype": "fp8",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "fp8",
  "moe_dtype": "fp8"
}
```

**Behavior**:
- GEMM: FP8 (W8A8 dynamic quantization)
- KV Cache: FP8
- Attention: FP8 (SGLang only, vLLM/TensorRT-LLM use GEMM dtype)
- MoE: FP8 (SGLang only, vLLM/TensorRT-LLM use GEMM dtype)

**Performance**: Memory: ~50%, Throughput: 1.5-2x (Hopper), Quality: Good

**Hardware**: Ampere (A100) or Hopper (H100), CUDA 11.8+

---

### Example 4: Aggressive MoE Quantization (SGLang Only)

```json
{
  "gemm_dtype": "bfloat16",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "fp8",
  "moe_dtype": "w4afp8"
}
```

**Behavior**:
- GEMM (non-MoE layers): BF16
- KV Cache: FP8
- Attention: FP8
- MoE experts: 4-bit weights + FP8 activations

**Performance**: Memory: ~40% (for MoE models), Throughput: 2-3x, Quality: Good

**Note**: Only works with SGLang. vLLM/TensorRT-LLM will use GEMM dtype for MoE.

---

### Example 5: Mixed Precision (Dense vs MoE)

```json
{
  "gemm_dtype": "float16",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "float16",
  "moe_dtype": "fp8"
}
```

**Behavior**:
- Dense layers: FP16 (high quality)
- KV Cache: FP8 (memory savings)
- Attention: FP16 (high quality)
- MoE experts: FP8 (memory + speed)

**Use Case**: MoE models where expert quality can be traded for speed, but attention quality is critical.

---

## Interaction with Offline Quantization

### Rule: Runtime quantization overrides GEMM dtype for offline models

**For offline-quantized models** (AWQ, GPTQ, GGUF, NVFP4):
- `gemm_dtype` is **ignored** (weights already quantized)
- `kvcache_dtype` **always applies** ✅
- `attention_dtype` behavior depends on engine:
  - vLLM: Falls back to model dtype
  - TensorRT-LLM: Uses model dtype
  - SGLang: Can override with FP8 attention
- `moe_dtype` behavior depends on engine:
  - vLLM/TensorRT-LLM: Uses model dtype
  - SGLang: Can override MoE backend

### Example: AWQ Model with Four-Field Config

```json
{
  "model": {"id_or_path": "TheBloke/Llama-2-7B-AWQ"},
  "quantization_config": {
    "gemm_dtype": "fp8",           // Ignored (AWQ uses INT4)
    "kvcache_dtype": "fp8_e5m2",   // Applied ✅
    "attention_dtype": "fp8",      // SGLang: Applied, vLLM/TRT: Ignored
    "moe_dtype": "auto"            // N/A (not MoE model)
  }
}
```

**Result**:
- Weights: AWQ INT4 (from model)
- KV Cache: FP8 (from runtime config)
- Attention: FP8 (SGLang only)

---

## Engine Compatibility Matrix

### GEMM Dtype Support

| Option | vLLM | TensorRT-LLM | SGLang | Notes |
|--------|------|--------------|--------|-------|
| `auto` | ✅ | ✅ | ✅ | Universal |
| `float16` | ✅ | ✅ | ✅ | Universal |
| `bfloat16` | ✅ | ✅ | ✅ | Universal |
| `float32` | ✅ | ⚠️ Limited | ❌ | vLLM only |
| `fp8` | ✅ | ✅ | ✅ | Ampere+ required |
| `int8` | ✅ | ✅ | ✅ | Universal |

### KV Cache Dtype Support

| Option | vLLM | TensorRT-LLM | SGLang | Notes |
|--------|------|--------------|--------|-------|
| `auto` | ✅ | ✅ | ✅ | Universal |
| `fp16` | ✅ | ✅ | ✅ | Universal |
| `bfloat16` | ✅ | ❌ | ✅ | TRT doesn't support explicit BF16 |
| `fp8` | ✅ | ✅ | ✅ | Ampere+ required |
| `fp8_e5m2` | ✅ | ✅ (as FP8) | ✅ | Ampere+ required |
| `fp8_e4m3` | ✅ | ✅ (as FP8) | ✅ | Ampere+ required |
| `int8` | ✅ | ✅ | ⚠️ Limited | Universal |
| `int4` | ❌ | ✅ | ❌ | TRT only |

### Attention Dtype Support

| Option | vLLM | TensorRT-LLM | SGLang | Notes |
|--------|------|--------------|--------|-------|
| `auto` | ✅ | ✅ | ✅ | Universal |
| `float16` | ✅ (via GEMM) | ✅ (default) | ✅ | Universal |
| `bfloat16` | ✅ (via GEMM) | ✅ (default) | ✅ | Universal |
| `fp8` | ❌ Falls back to GEMM | ✅ `--fmha-quant-algo FP8` | ✅ | **TRT-LLM and SGLang** |
| `fp8_e5m2` | ❌ | ✅ (as FP8) | ✅ | **TRT-LLM and SGLang** |
| `fp8_e4m3` | ❌ | ✅ (as FP8) | ✅ | **TRT-LLM and SGLang** |
| `fp8_block` | ❌ | ✅ `FP8_BLOCK` | ⚠️ Experimental | TRT-LLM only |

### MoE Dtype Support

| Option | vLLM | TensorRT-LLM | SGLang | Notes |
|--------|------|--------------|--------|-------|
| `auto` | ✅ | ✅ | ✅ | Universal |
| `float16` | ✅ (via GEMM) | ✅ (via GEMM) | ✅ | Universal |
| `bfloat16` | ✅ (via GEMM) | ✅ (via GEMM) | ✅ | Universal |
| `fp8` | ⚠️ Limited | ❌ Uses GEMM | ✅ | SGLang best support |
| `w4afp8` | ❌ | ❌ | ✅ | SGLang only |
| `mxfp4` | ❌ | ❌ | ✅ | SGLang + Blackwell only |
| `int8` | ⚠️ Limited | ❌ | ✅ | SGLang best support |

**Legend**:
- ✅ Full support
- ⚠️ Partial support (limited features or experimental)
- ❌ Not supported (falls back to default)

---

## Hardware Requirements

| Feature | Pascal (GTX 10xx) | Volta (V100) | Ampere (A100) | Hopper (H100) | Blackwell (B100/B200) |
|---------|-------------------|--------------|---------------|---------------|-----------------------|
| FP16/BF16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 (all fields) | ❌ | ❌ | ✅ | ✅ (2x accel) | ✅ (2x accel) |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| MoE W4A8 | ❌ | ❌ | ✅ | ✅ | ✅ |
| MXFP4 | ❌ | ❌ | ❌ | ❌ | ✅ (4x accel) |

**CUDA Requirements**:
- FP8: CUDA 11.8+
- MXFP4: CUDA 12.8+, PyTorch 2.8+

---

## Summary

### Key Design Points

1. **Four Orthogonal Fields**: GEMM, KV Cache, Attention, MoE
2. **Each field controls a specific computation component**
3. **Runtime only**: Offline quantization auto-detected from model
4. **Three-engine universal**: All fields work across vLLM, TensorRT-LLM, SGLang (with varying support levels)

### Engine Comparison

| Feature | vLLM | TensorRT-LLM | SGLang | Winner |
|---------|------|--------------|--------|--------|
| GEMM dtype control | ✅ Full | ✅ Full | ✅ Full | Tie |
| KV cache dtype control | ✅ Full | ✅ Full (+ INT4) | ✅ Full | **TRT-LLM** (INT4 support) |
| Attention dtype control | ❌ Limited | ✅ **FMHA quant** | ✅ **Full** | **TRT-LLM & SGLang** |
| MoE dtype control | ⚠️ Limited | ❌ No | ✅ **Full** | **SGLang** |
| Overall flexibility | Good | **Excellent** | **Excellent** | **TRT-LLM & SGLang** |

### Recommendation

**For most users** (dense models):
```json
{
  "gemm_dtype": "auto",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "auto",
  "moe_dtype": "auto"
}
```

**For MoE models with SGLang**:
```json
{
  "gemm_dtype": "bfloat16",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "fp8",
  "moe_dtype": "w4afp8"
}
```

**For maximum throughput (Hopper GPU)**:
```json
{
  "gemm_dtype": "fp8",
  "kvcache_dtype": "fp8_e5m2",
  "attention_dtype": "fp8",
  "moe_dtype": "fp8"
}
```
