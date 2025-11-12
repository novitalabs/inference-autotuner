# Quantization Configuration Quick Reference

## TL;DR

Add `quant_config` to your task JSON for runtime quantization control:

```json
{
  "quant_config": {"preset": "kv-cache-fp8"}
}
```

## Four Fields

| Field | Controls | Options |
|-------|----------|---------|
| `gemm_dtype` | Linear layers, MLPs | `auto`, `fp8`, `int8`, `float16`, `bfloat16` |
| `kvcache_dtype` | KV cache storage | `auto`, `fp8_e5m2`, `fp8_e4m3`, `int8`, `int4` |
| `attention_dtype` | Attention computation | `auto`, `fp8`, `fp8_e5m2`, `fp8_block` |
| `moe_dtype` | MoE experts | `auto`, `fp8`, `w4afp8`, `mxfp4` |

## Presets

```json
"default"          // No quantization
"kv-cache-fp8"     // ⭐ RECOMMENDED: FP8 KV cache only
"dynamic-fp8"      // Full FP8 (Hopper GPU)
"bf16-stable"      // BF16 + FP8 KV cache
"aggressive-moe"   // MoE quantization (SGLang)
```

## Examples

### Preset (Easy)
```json
{"quant_config": {"preset": "kv-cache-fp8"}}
```

### Custom (Advanced)
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

### Compare Multiple
```json
{
  "quant_config": {
    "presets": ["default", "kv-cache-fp8", "dynamic-fp8"]
  }
}
```

## Engine Support

| Feature | vLLM | TRT-LLM | SGLang |
|---------|------|---------|--------|
| GEMM | ✅ | ✅ | ✅ |
| KV Cache | ✅ | ✅ (+INT4) | ✅ |
| Attention | ❌ | ✅ FMHA | ✅ |
| MoE | ⚠️ | ❌ | ✅ |

## Best Practices

🎯 **Most users**: `"preset": "kv-cache-fp8"` (25-50% memory, <0.1% quality loss)

🚀 **Hopper GPU**: `"preset": "dynamic-fp8"` (50% memory, 2x throughput)

🧠 **MoE models**: Use SGLang with `"moe_dtype": "w4afp8"`

## Parameter Priority

User params > quant_config

```json
{
  "quant_config": {"kvcache_dtype": "fp8_e5m2"},
  "parameters": {"kv-cache-dtype": "int8"}  // This wins
}
```

## Offline Models (AWQ/GPTQ)

✅ KV cache quantization still works
❌ gemm_dtype ignored (model already quantized)

## Docs

- Usage Guide: `docs/QUANTIZATION_USAGE.md`
- Technical Spec: `docs/QUANTIZATION_FOUR_FIELDS.md`
- Implementation: `docs/QUANTIZATION_IMPLEMENTATION_SUMMARY.md`

## Tests

```bash
PYTHONPATH=src python tests/test_quantization_mapper.py
```
