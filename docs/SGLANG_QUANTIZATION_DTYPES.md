# SGLang Quantization and DType Parameters

This document provides a comprehensive overview of quantization methods and dtype parameters supported by SGLang for LLM inference optimization.

## Overview

SGLang supports two main categories of quantization:
1. **Offline Quantization**: Pre-quantized model weights loaded directly during inference
2. **Online Quantization**: Dynamic quantization computed at runtime

**Recommendation**: Offline quantization is preferred for better performance, usability, and convenience.

## Base Data Types (--dtype)

The `--dtype` parameter controls the precision of model weights and activations.

### Supported Values

| Value | Description | Use Case |
|-------|-------------|----------|
| `auto` | Automatic selection: FP16 for FP32/FP16 models, BF16 for BF16 models | Default, recommended |
| `half` / `float16` | FP16 (16-bit floating point) | Recommended for AWQ quantization |
| `bfloat16` | BF16 (Brain Float 16) | Better numerical stability than FP16 |
| `float` / `float32` | FP32 (32-bit floating point) | Full precision, slower but highest accuracy |

### Example
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --dtype bfloat16 \
    --port 30000
```

## Quantization Methods (--quantization)

The `--quantization` parameter specifies the quantization technique to use.

### All Supported Methods

From `server_args.py`, SGLang supports the following 21 quantization methods:

```python
QUANTIZATION_CHOICES = [
    "awq",              # Activation-aware Weight Quantization (4-bit)
    "fp8",              # 8-bit floating point
    "gptq",             # GPT Quantization (various bit-widths)
    "marlin",           # Optimized GPTQ/AWQ kernel
    "gptq_marlin",      # GPTQ with Marlin kernel
    "awq_marlin",       # AWQ with Marlin kernel
    "bitsandbytes",     # 8-bit and 4-bit quantization from bitsandbytes library
    "gguf",             # GGUF format (llama.cpp compatible)
    "modelopt",         # NVIDIA ModelOpt quantization
    "modelopt_fp8",     # ModelOpt FP8 quantization
    "modelopt_fp4",     # ModelOpt FP4 quantization (Blackwell GPUs)
    "petit_nvfp4",      # NVIDIA FP4 quantization
    "w8a8_int8",        # 8-bit weights, 8-bit activations (INT8)
    "w8a8_fp8",         # 8-bit weights, 8-bit activations (FP8)
    "moe_wna16",        # MoE with 16-bit activations
    "qoq",              # Quantization-on-Quantization
    "w4afp8",           # 4-bit weights with FP8 activations
    "mxfp4",            # Microscaling FP4 format
    "auto-round",       # Intel Auto-Round quantization
    "compressed-tensors" # Ktransformers compressed format
]
```

### Detailed Method Descriptions

#### Offline Quantization Methods

**1. AWQ (Activation-aware Weight Quantization)**
- 4-bit weight quantization
- Preserves important weights based on activation statistics
- Good balance between size and accuracy
```bash
# Load pre-quantized AWQ model (no --quantization flag needed)
python -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

**2. GPTQ**
- Flexible bit-width quantization (2/3/4/8-bit)
- Calibration-based method
- Requires pre-quantized model
```bash
# Load GPTQ model
python -m sglang.launch_server \
    --model-path TheBloke/Llama-2-7B-GPTQ
```

**3. Marlin Variants (marlin, gptq_marlin, awq_marlin)**
- Optimized GPU kernels for GPTQ/AWQ
- Better performance than standard implementations
- Use for pre-quantized GPTQ/AWQ models

**4. GGUF**
- llama.cpp compatible format
- Multiple quantization levels (Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.)
- Cross-platform compatibility
```bash
# GGUF format auto-detected from model config
python -m sglang.launch_server \
    --model-path TheBloke/Llama-2-7B-GGUF
```

**5. ModelOpt (NVIDIA)**
- Enterprise-grade quantization from NVIDIA
- Three variants:
  - `modelopt`: General ModelOpt quantization
  - `modelopt_fp8`: FP8 quantization (Hopper/Blackwell GPUs)
  - `modelopt_fp4`: FP4 quantization (Blackwell GPUs only)

```bash
# After quantizing with ModelOpt
python -m sglang.launch_server \
    --model-path ./quantized_tinyllama_fp8 \
    --quantization modelopt
```

**6. W8A8 Variants**
- `w8a8_int8`: 8-bit weights + 8-bit INT8 activations
- `w8a8_fp8`: 8-bit weights + 8-bit FP8 activations
- Optimized CUTLASS kernels via sgl-kernel

```bash
# Override config for optimized kernel
python -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
    --quantization w8a8_fp8
```

**7. auto-round (Intel)**
- Supports multiple schemes: W2A16, W3A16, W4A16, W8A16, NVFP4, MXFP4, GGUF
- Works on Gaudi/CPU/Intel GPU/CUDA
```bash
# Auto-round model (quantization detected from config)
python -m sglang.launch_server \
    --model-path ./Llama-3.2-1B-Instruct-autoround-4bit
```

**8. compressed-tensors**
- Ktransformers format
- Space-efficient tensor compression

#### Online Quantization Methods

**1. FP8 (Online)**
- 8-bit floating point quantization
- Dynamic scaling computed at runtime
- Good performance on modern NVIDIA GPUs
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8
```

**2. bitsandbytes**
- 8-bit and 4-bit quantization from bitsandbytes library
- Memory-efficient for consumer GPUs
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --quantization bitsandbytes
```

**3. Torchao Integration (--torchao-config)**
- Alternative quantization framework from PyTorch
- Supported methods: `["int8dq", "int8wo", "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row", "int4wo-32", "int4wo-64", "int4wo-128", "int4wo-256"]`

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --torchao-config int4wo-128
```

**Note**: `int8dq` method requires `--disable-cuda-graph` due to compatibility issues.

## KV Cache Data Type (--kv-cache-dtype)

Controls the precision of key-value cache storage, which can significantly reduce memory usage.

### Supported Values

| Value | Description | Requirements |
|-------|-------------|--------------|
| `auto` | Use model data type (default) | All GPUs |
| `fp8_e5m2` | FP8 E5M2 format | CUDA 11.8+ |
| `fp8_e4m3` | FP8 E4M3 format | CUDA 11.8+ |
| `bf16` / `bfloat16` | BF16 KV cache | All GPUs |
| `fp4_e2m1` | FP4 (mxfp4 only) | CUDA 12.8+, PyTorch 2.8+ |

### Example
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --kv-cache-dtype fp8_e5m2
```

### Memory Savings
- FP8 KV cache: ~50% memory reduction vs FP16
- FP4 KV cache: ~75% memory reduction vs FP16
- Critical for long context windows and batch processing

## ModelOpt Advanced Parameters

For NVIDIA ModelOpt quantization workflow:

### --modelopt-quant
Specifies ModelOpt quantization configuration.

**Supported values**: `fp8`, `int4_awq`, `w4a8_awq`, `nvfp4`, `nvfp4_awq`

**Requirements**: `pip install nvidia-modelopt`

```bash
# Quantize and serve immediately
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --modelopt-quant fp8 \
    --quantize-and-serve
```

### --modelopt-checkpoint-save-path
Path to save fake quantized checkpoint after quantization.
Allows reusing calibration results in future runs.

### --modelopt-checkpoint-restore-path
Path to restore a previously saved ModelOpt checkpoint.
Skips quantization process if checkpoint exists.

### --modelopt-export-path
Path to export quantized model in HuggingFace format.
Enables deployment without re-quantization.

### --quantize-and-serve
Quantize the model with ModelOpt and immediately serve it without exporting.

## Usage Recommendations

### For Production Deployments

1. **High Performance (H100/B100 GPUs)**:
   - Use `modelopt_fp8` or offline FP8 models
   - Enable `--kv-cache-dtype fp8_e5m2`

2. **Memory Constrained**:
   - Use AWQ or GPTQ 4-bit quantization
   - Enable `--kv-cache-dtype fp8_e5m2` or `fp4_e2m1`

3. **Best Accuracy**:
   - Use `--dtype bfloat16` without quantization
   - Or use offline FP8 quantization with proper calibration

### For Development/Testing

1. **Quick Testing**:
   ```bash
   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-1B-Instruct \
       --dtype auto
   ```

2. **Memory-Efficient Testing**:
   ```bash
   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-1B-Instruct \
       --quantization fp8 \
       --kv-cache-dtype fp8_e5m2
   ```

## Autotuner Parameter Examples

For inference-autotuner task configurations, you can tune these quantization parameters:

### Example 1: Tuning KV Cache DType
```json
{
  "parameters": {
    "tp-size": [1, 2],
    "kv-cache-dtype": ["auto", "fp8_e5m2", "fp8_e4m3"]
  }
}
```

### Example 2: Tuning with Pre-Quantized Model
```json
{
  "model": {
    "id_or_path": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
  },
  "parameters": {
    "quantization": ["w8a8_fp8"],
    "tp-size": [1, 2, 4],
    "mem-fraction-static": [0.85, 0.9]
  }
}
```

### Example 3: Tuning ModelOpt Quantization
```json
{
  "model": {
    "id_or_path": "meta-llama/Llama-3.2-1B-Instruct"
  },
  "parameters": {
    "modelopt-quant": ["fp8"],
    "tp-size": [1, 2],
    "enable-fp32-lm-head": [true, false]
  }
}
```

## Important Notes

### Pre-Quantized Models
- **DO NOT** add `--quantization` flag when loading pre-quantized models
- Quantization method is auto-detected from HuggingFace config
- Exception: Can override with `w8a8_int8` or `w8a8_fp8` for optimized kernels

### Model-Specific Considerations

**DeepSeek V3/R1**:
- Already quantized to FP8 natively
- Do not add redundant quantization parameters

**MoE Models**:
- Limited support for quantized MoE
- May need to skip mlp.gate layer quantization

**VLMs (Vision-Language Models)**:
- Limited quantization support
- AWQ and auto_awq formats work best
- GPTQ may have compatibility issues

### Compatibility Matrix

| Method | CUDA Version | PyTorch Version | GPU Requirements |
|--------|--------------|-----------------|------------------|
| FP8 (most) | CUDA 11.8+ | Any | Ampere or newer |
| FP4 (mxfp4) | CUDA 12.8+ | PyTorch 2.8+ | Hopper or newer |
| ModelOpt FP8 | CUDA 11.8+ | Any | Hopper/Blackwell |
| ModelOpt FP4 | CUDA 12.0+ | Any | Blackwell only |
| AWQ/GPTQ | CUDA 11.4+ | Any | Pascal or newer |

## References

- [SGLang Quantization Documentation](https://sgl-project.github.io/docs/advanced_features/quantization.html)
- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor/)
- [Intel Auto-Round](https://github.com/intel/auto-round)
- [PyTorch Torchao](https://github.com/pytorch/ao)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/)
