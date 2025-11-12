"""
Unit tests for quantization mapper.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.quantization_mapper import (
    map_to_vllm_args,
    map_to_sglang_args,
    map_to_tensorrt_llm_args,
    resolve_quant_config,
    merge_parameters,
    get_runtime_args
)


def test_resolve_preset():
    """Test preset expansion."""
    print("Test 1: Resolve preset")
    config = {"preset": "kv-cache-fp8"}
    resolved = resolve_quant_config(config)
    print(f"  Input: {config}")
    print(f"  Resolved: {resolved}")
    assert resolved["gemm_dtype"] == "auto"
    assert resolved["kvcache_dtype"] == "fp8_e5m2"
    print("  ✓ PASSED\n")


def test_vllm_mapping():
    """Test vLLM argument mapping."""
    print("Test 2: vLLM mapping")
    config = {
        "gemm_dtype": "fp8",
        "kvcache_dtype": "fp8_e5m2",
        "attention_dtype": "auto",
        "moe_dtype": "auto"
    }
    args = map_to_vllm_args(config)
    print(f"  Input: {config}")
    print(f"  Output: {args}")
    assert args["--quantization"] == "fp8"
    assert args["--kv-cache-dtype"] == "fp8_e5m2"
    print("  ✓ PASSED\n")


def test_sglang_mapping():
    """Test SGLang argument mapping."""
    print("Test 3: SGLang mapping")
    config = {
        "gemm_dtype": "fp8",
        "kvcache_dtype": "fp8_e5m2",
        "attention_dtype": "fp8",
        "moe_dtype": "auto"
    }
    args = map_to_sglang_args(config)
    print(f"  Input: {config}")
    print(f"  Output: {args}")
    assert args["--quantization"] == "fp8"
    assert args["--kv-cache-dtype"] == "fp8_e5m2"
    assert args["--attention-backend"] == "flashinfer"
    print("  ✓ PASSED\n")


def test_tensorrt_llm_mapping():
    """Test TensorRT-LLM argument mapping."""
    print("Test 4: TensorRT-LLM mapping")
    config = {
        "gemm_dtype": "fp8",
        "kvcache_dtype": "fp8_e5m2",
        "attention_dtype": "fp8",
        "moe_dtype": "auto"
    }
    args = map_to_tensorrt_llm_args(config)
    print(f"  Input: {config}")
    print(f"  Output: {args}")
    assert args["--quant-algo"] == "FP8"
    assert args["--kv-cache-quant-algo"] == "FP8"
    assert args["--fmha-quant-algo"] == "FP8"
    print("  ✓ PASSED\n")


def test_parameter_merge():
    """Test parameter merging with user overrides."""
    print("Test 5: Parameter merge")
    quant_args = {
        "--dtype": "auto",
        "--kv-cache-dtype": "fp8_e5m2"
    }
    user_params = {
        "tp-size": 2,
        "mem-fraction-static": 0.85,
        "kv-cache-dtype": "fp8_e4m3"  # Override quant_config
    }
    merged = merge_parameters(quant_args, user_params)
    print(f"  Quant args: {quant_args}")
    print(f"  User params: {user_params}")
    print(f"  Merged: {merged}")
    assert merged["--kv-cache-dtype"] == "fp8_e4m3"  # User override
    assert merged["--tp-size"] == "2"
    assert merged["--mem-fraction-static"] == "0.85"
    print("  ✓ PASSED\n")


def test_end_to_end():
    """Test end-to-end workflow."""
    print("Test 6: End-to-end workflow")
    quant_config = {"preset": "dynamic-fp8"}
    user_params = {"tp-size": 2, "mem-fraction-static": 0.85}

    args = get_runtime_args(
        runtime="sglang",
        quant_config=quant_config,
        user_parameters=user_params
    )
    print(f"  Quant config: {quant_config}")
    print(f"  User params: {user_params}")
    print(f"  Final args: {args}")
    assert "--quantization" in args
    assert args["--quantization"] == "fp8"
    assert args["--tp-size"] == "2"
    print("  ✓ PASSED\n")


def test_offline_quantization_detection():
    """Test offline quantization detection and fallback."""
    print("Test 7: Offline quantization detection")
    config = {
        "gemm_dtype": "fp8",  # This should be ignored for AWQ model
        "kvcache_dtype": "fp8_e5m2",
        "attention_dtype": "auto",
        "moe_dtype": "auto"
    }
    # Simulate AWQ model
    args = map_to_vllm_args(config, model_quantization="awq")
    print(f"  Input: {config}")
    print(f"  Model quantization: awq")
    print(f"  Output: {args}")
    # gemm_dtype='fp8' should be ignored (model already quantized)
    assert "--quantization" not in args or args["--quantization"] != "fp8"
    assert args["--kv-cache-dtype"] == "fp8_e5m2"  # KV cache still applied
    print("  ✓ PASSED\n")


def test_moe_quantization():
    """Test MoE-specific quantization for SGLang."""
    print("Test 8: MoE quantization (SGLang)")
    config = {
        "gemm_dtype": "bfloat16",
        "kvcache_dtype": "fp8_e5m2",
        "attention_dtype": "fp8",
        "moe_dtype": "w4afp8"
    }
    args = map_to_sglang_args(config)
    print(f"  Input: {config}")
    print(f"  Output: {args}")
    assert args["--quantization"] == "w4afp8"  # MoE overrides GEMM
    assert args["--moe-runner-backend"] == "flashinfer_mxfp4"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantization Mapper Unit Tests")
    print("=" * 60 + "\n")

    try:
        test_resolve_preset()
        test_vllm_mapping()
        test_sglang_mapping()
        test_tensorrt_llm_mapping()
        test_parameter_merge()
        test_end_to_end()
        test_offline_quantization_detection()
        test_moe_quantization()

        print("=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
