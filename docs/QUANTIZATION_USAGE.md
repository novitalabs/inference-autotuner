# Quantization Configuration Guide

## Overview

The autotuner now supports **four-field runtime quantization configuration** for fine-grained control over model precision. This allows you to independently configure:

1. **GEMM dtype**: Computation precision for matrix multiplications (linear layers, MLPs)
2. **KV Cache dtype**: Storage precision for key-value cache tensors
3. **Attention dtype**: Computation precision for attention mechanisms (FMHA)
4. **MoE dtype**: Computation precision for Mixture-of-Experts layers

## Quick Start

### Using Presets (Recommended)

The easiest way to use quantization is with built-in presets:

```json
{
  "quant_config": {
    "preset": "kv-cache-fp8"
  }
}
```

**Available Presets**:
- `default`: No runtime quantization (baseline)
- `kv-cache-fp8`: FP8 KV cache only (recommended, minimal quality impact)
- `dynamic-fp8`: Full FP8 (GEMM + KV cache + Attention)
- `bf16-stable`: BF16 computation with FP8 KV cache
- `aggressive-moe`: Aggressive MoE quantization (W4A8, SGLang only)

### Custom Configuration

For fine-grained control, specify each field explicitly:

```json
{
  "quant_config": {
    "gemm_dtype": "fp8",
    "kvcache_dtype": "fp8_e5m2",
    "attention_dtype": "fp8",
    "moe_dtype": "auto"
  }
}
```

### Multi-Preset Comparison

Compare multiple quantization strategies in a single task:

```json
{
  "quant_config": {
    "presets": ["default", "kv-cache-fp8", "dynamic-fp8"]
  }
}
```

The autotuner will create experiments for each preset.

## Field Options

### gemm_dtype (GEMM Computation Precision)

Controls precision for linear layers and MLPs.

**Options**: `auto`, `float16`, `bfloat16`, `float32`, `fp8`, `int8`

- `auto`: Follow model default (recommended)
- `fp8`: Dynamic W8A8 quantization (weights + activations)
- `int8`: INT8 quantization

**Note**: `fp8` and `int8` only apply to **unquantized models**. For offline-quantized models (AWQ, GPTQ, GGUF), this field is ignored.

### kvcache_dtype (KV Cache Storage Precision)

Controls storage precision for key-value cache tensors.

**Options**: `auto`, `fp16`, `bfloat16`, `fp8`, `fp8_e5m2`, `fp8_e4m3`, `int8`, `int4`

- `fp8_e5m2`: Best quality (5-bit exponent)
- `fp8_e4m3`: Best hardware compatibility (4-bit exponent)
- `int4`: Extreme compression (TensorRT-LLM only)

**Memory Savings**: ~50% for FP8, ~75% for INT4

### attention_dtype (Attention Computation Precision)

Controls computation precision for attention mechanisms.

**Options**: `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e5m2`, `fp8_e4m3`, `fp8_block`

- `fp8`: FP8 attention (FMHA)
- `fp8_block`: Block-wise FP8 (TensorRT-LLM experimental)

**Engine Support**:
- ✅ **TensorRT-LLM**: Full support (`--fmha-quant-algo`)
- ✅ **SGLang**: Full support (with FlashInfer backend)
- ❌ **vLLM**: Falls back to GEMM dtype

### moe_dtype (MoE Expert Computation Precision)

Controls computation precision for Mixture-of-Experts layers.

**Options**: `auto`, `float16`, `bfloat16`, `fp8`, `w4afp8`, `mxfp4`, `int8`

- `fp8`: FP8 experts (W8A8)
- `w4afp8`: 4-bit weights + FP8 activations (SGLang only)
- `mxfp4`: MXFP4 (Blackwell GPU, SGLang only)

**Engine Support**:
- ✅ **SGLang**: Full support
- ⚠️ **vLLM**: Limited support
- ❌ **TensorRT-LLM**: Uses GEMM dtype

## Parameter Priority

User-specified parameters in `parameters` field **always override** quant_config-derived values:

```json
{
  "quant_config": {
    "kvcache_dtype": "fp8_e5m2"
  },
  "parameters": {
    "kv-cache-dtype": "fp8_e4m3",  // This takes priority
    "tp-size": [1, 2]
  }
}
```

Result: `kv-cache-dtype` will be `fp8_e4m3`, not `fp8_e5m2`.

## Engine-Specific Mapping

### vLLM

```python
# quant_config
{"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"}

# Mapped to vLLM args
{"--quantization": "fp8", "--dtype": "auto", "--kv-cache-dtype": "fp8_e5m2"}
```

**Limitations**:
- No separate attention dtype
- Limited MoE dtype control

### SGLang

```python
# quant_config
{"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "fp8"}

# Mapped to SGLang args
{
  "--quantization": "fp8",
  "--dtype": "auto",
  "--kv-cache-dtype": "fp8_e5m2",
  "--attention-backend": "flashinfer"
}
```

**Advantages**:
- Full attention dtype control
- Full MoE dtype control

### TensorRT-LLM

```python
# quant_config
{"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "fp8"}

# Mapped to TensorRT-LLM args
{
  "--quant-algo": "FP8",
  "--kv-cache-quant-algo": "FP8",
  "--fmha-quant-algo": "FP8"
}
```

**Unique Features**:
- INT4 KV cache support
- FMHA quantization (`--fmha-quant-algo`)

## Examples

See `examples/` directory:

1. **quant_preset_task.json**: Using preset (`kv-cache-fp8`)
2. **quant_custom_task.json**: Custom four-field configuration
3. **quant_multi_preset_task.json**: Multi-preset comparison

## Offline Quantization Detection

The autotuner automatically detects offline-quantized models (AWQ, GPTQ, GGUF, NVFP4) and:

1. **Ignores gemm_dtype**: Model weights are already quantized
2. **Applies kvcache_dtype**: KV cache quantization is always runtime-configurable
3. **Logs warning**: If you try to apply FP8 to an already-quantized model

Example with AWQ model:

```json
{
  "model": {
    "id_or_path": "TheBloke/Llama-2-7B-AWQ"
  },
  "quant_config": {
    "gemm_dtype": "fp8",           // Ignored (AWQ already 4-bit)
    "kvcache_dtype": "fp8_e5m2"    // Applied ✓
  }
}
```

Result:
- Weights: AWQ 4-bit (from model)
- KV Cache: FP8 (from runtime config)

## Best Practices

### For Most Users (Dense Models)

Use `kv-cache-fp8` preset:

```json
{
  "quant_config": {
    "preset": "kv-cache-fp8"
  }
}
```

**Benefits**:
- 25-50% memory savings
- Minimal quality impact (<0.1% degradation)
- Universal engine support

### For Hopper GPUs (H100)

Use `dynamic-fp8` preset:

```json
{
  "quant_config": {
    "preset": "dynamic-fp8"
  }
}
```

**Benefits**:
- ~50% memory savings
- 1.5-2x throughput improvement
- Small quality impact (~0.5% degradation)

### For MoE Models (SGLang)

Use `aggressive-moe` or custom MoE configuration:

```json
{
  "quant_config": {
    "gemm_dtype": "bfloat16",
    "kvcache_dtype": "fp8_e5m2",
    "attention_dtype": "fp8",
    "moe_dtype": "w4afp8"
  }
}
```

### For Numerical Stability

Use `bf16-stable` preset:

```json
{
  "quant_config": {
    "preset": "bf16-stable"
  }
}
```

## Hardware Requirements

| Feature | Pascal | Volta | Ampere | Hopper | Blackwell |
|---------|--------|-------|--------|--------|-----------|
| FP16/BF16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 | ❌ | ❌ | ✅ | ✅ (2x) | ✅ (2x) |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| MXFP4 | ❌ | ❌ | ❌ | ❌ | ✅ (4x) |

**CUDA Requirements**:
- FP8: CUDA 11.8+
- MXFP4: CUDA 12.8+

## Troubleshooting

### Warning: "Ignoring gemm_dtype='fp8': model already quantized"

**Cause**: You're trying to apply dynamic FP8 to an offline-quantized model (AWQ, GPTQ, etc.)

**Solution**: Remove `gemm_dtype` or set it to `auto`. KV cache quantization will still work.

### vLLM: "Attention dtype not supported"

**Cause**: vLLM doesn't support separate attention dtype.

**Solution**: Use SGLang or TensorRT-LLM for attention dtype control.

### Parameter not applied

**Cause**: User parameters in `parameters` field override quant_config.

**Solution**: Check your `parameters` field for conflicting values.

## API Integration

### Python API

```python
from utils.quantization_mapper import get_runtime_args

# Get runtime arguments
args = get_runtime_args(
    runtime="sglang",
    quant_config={"preset": "kv-cache-fp8"},
    user_parameters={"tp-size": 2}
)

# Result: {"--quantization": ..., "--tp-size": "2", ...}
```

### Task Creation

When creating tasks via REST API, include `quant_config`:

```python
import requests

task_data = {
    "task_name": "my-quantization-task",
    "model": {...},
    "quant_config": {"preset": "kv-cache-fp8"},
    "parameters": {...}
}

response = requests.post("http://localhost:8000/api/tasks/", json=task_data)
```

## Reference

- **Documentation**: `docs/QUANTIZATION_FOUR_FIELDS.md`
- **Implementation**: `src/utils/quantization_mapper.py`
- **Tests**: `tests/test_quantization_mapper.py`
- **Migration**: `migrations/001_add_quant_config.py`

## Support Matrix

| Engine | GEMM | KV Cache | Attention | MoE |
|--------|------|----------|-----------|-----|
| **vLLM** | ✅ Full | ✅ Full | ❌ Limited | ⚠️ Limited |
| **TensorRT-LLM** | ✅ Full | ✅ Full (+ INT4) | ✅ **FMHA** | ❌ No |
| **SGLang** | ✅ Full | ✅ Full | ✅ **Full** | ✅ **Full** |

**Winner**: **SGLang** has the most comprehensive quantization support, especially for attention and MoE models.
