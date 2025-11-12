"""
Integration helper for quantization configuration in orchestrator.

This module provides utilities to integrate quantization configuration
with the autotuner orchestrator and parameter grid expansion.
"""

import logging
import itertools
from typing import Dict, Any, Optional, List

from utils.quantization_mapper import (
    get_runtime_args,
    resolve_quant_config,
    validate_quant_config
)

logger = logging.getLogger(__name__)


def detect_model_quantization(model_path: str, model_config: Optional[Dict] = None) -> Optional[str]:
    """
    Detect offline quantization method from model path or config.

    Args:
        model_path: Model path or HuggingFace model ID
        model_config: Optional model configuration dict

    Returns:
        Quantization method name ("awq", "gptq", "gguf", etc.) or None if not quantized
    """
    # Check model path for quantization indicators
    model_path_lower = model_path.lower()

    # Common quantization patterns in model names
    if "awq" in model_path_lower:
        return "awq"
    elif "gptq" in model_path_lower:
        return "gptq"
    elif "gguf" in model_path_lower or model_path.endswith(".gguf"):
        return "gguf"
    elif "nvfp4" in model_path_lower or "fp4" in model_path_lower:
        return "nvfp4"
    elif "fp8" in model_path_lower:
        return "fp8"

    # Check model_config for quantization_config field
    if model_config and isinstance(model_config, dict):
        if "quantization" in model_config:
            return model_config["quantization"]
        if "quantization_config" in model_config:
            quant_config = model_config["quantization_config"]
            if isinstance(quant_config, dict) and "quant_method" in quant_config:
                return quant_config["quant_method"]

    return None


def prepare_experiment_parameters(
    base_runtime: str,
    quant_config: Optional[Dict[str, Any]],
    param_combination: Dict[str, Any],
    model_path: str,
    model_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Prepare final experiment parameters by merging quant_config and user parameters.

    Args:
        base_runtime: Runtime engine ("vllm", "sglang", "tensorrt_llm")
        quant_config: Quantization configuration from task
        param_combination: Single parameter combination from grid (user parameters)
        model_path: Model path for detecting offline quantization
        model_config: Optional model configuration

    Returns:
        Merged parameter dictionary with quant_config-derived args + user overrides

    Example:
        >>> prepare_experiment_parameters(
        ...     "sglang",
        ...     {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"},
        ...     {"tp-size": 2, "mem-fraction-static": 0.85},
        ...     "meta-llama/Llama-3.2-1B-Instruct"
        ... )
        {"quantization": "fp8", "dtype": "auto", "kv-cache-dtype": "fp8_e5m2",
         "tp-size": 2, "mem-fraction-static": 0.85}
    """
    # Detect offline model quantization
    model_quantization = detect_model_quantization(model_path, model_config)

    if model_quantization:
        logger.info(f"Detected offline model quantization: {model_quantization}")

    # Get runtime arguments from quant_config
    try:
        runtime_args = get_runtime_args(
            runtime=base_runtime,
            quant_config=quant_config,
            user_parameters=param_combination,
            model_quantization=model_quantization
        )
    except ValueError as e:
        logger.error(f"Failed to process quantization config: {e}")
        # Fall back to user parameters only
        runtime_args = param_combination.copy()

    # Remove CLI prefix (--) for cleaner parameter dict
    # Controllers will add it back when building commands
    cleaned_params = {}
    for key, value in runtime_args.items():
        clean_key = key.lstrip("-")
        cleaned_params[clean_key] = value

    return cleaned_params


def expand_quantization_presets(
    quant_config: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Expand quantization configuration if it contains multiple presets.

    Args:
        quant_config: Quantization config (can contain "presets" list)

    Returns:
        List of expanded quantization configs (one per preset)

    Example:
        >>> expand_quantization_presets({"presets": ["default", "kv-cache-fp8"]})
        [
            {"gemm_dtype": "auto", "kvcache_dtype": "auto", ...},
            {"gemm_dtype": "auto", "kvcache_dtype": "fp8_e5m2", ...}
        ]
    """
    if not quant_config:
        return [None]

    # Check if multiple presets are specified
    if "presets" in quant_config:
        presets = quant_config["presets"]
        if not isinstance(presets, list):
            presets = [presets]

        expanded = []
        for preset_name in presets:
            preset_config = {"preset": preset_name}
            # Allow overriding preset values with explicit fields
            preset_config.update({k: v for k, v in quant_config.items() if k != "presets"})
            expanded.append(preset_config)

        return expanded

    # Single configuration
    return [quant_config]


def validate_task_quant_config(quant_config: Optional[Dict[str, Any]]) -> tuple[bool, str]:
    """
    Validate quantization configuration from task.

    Args:
        quant_config: Quantization configuration to validate

    Returns:
        (is_valid, error_message) tuple
    """
    if not quant_config:
        return True, "Valid (no quantization config)"

    return validate_quant_config(quant_config)


def get_quant_config_summary(quant_config: Optional[Dict[str, Any]]) -> str:
    """
    Get a human-readable summary of quantization configuration.

    Args:
        quant_config: Quantization configuration

    Returns:
        Summary string

    Example:
        >>> get_quant_config_summary({"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"})
        "GEMM: fp8, KV Cache: fp8_e5m2, Attention: auto, MoE: auto"
    """
    if not quant_config:
        return "No quantization"

    # Handle preset references
    if "preset" in quant_config:
        resolved = resolve_quant_config(quant_config)
    else:
        resolved = quant_config

    gemm = resolved.get("gemm_dtype", "auto")
    kvcache = resolved.get("kvcache_dtype", "auto")
    attention = resolved.get("attention_dtype", "auto")
    moe = resolved.get("moe_dtype", "auto")

    return f"GEMM: {gemm}, KV Cache: {kvcache}, Attention: {attention}, MoE: {moe}"


def expand_quant_config_to_parameter_spec(
    quant_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert quant_config with arrays into parameter spec format for generate_parameter_grid.

    This function handles quant_config that contains arrays of values and converts them
    into the format expected by generate_parameter_grid (simple list format).

    Args:
        quant_config: Quantization config with possible arrays
                     Example: {"gemm_dtype": ["auto", "fp8"], "kvcache_dtype": ["auto", "fp8_e5m2"]}

    Returns:
        Parameter spec dict in generate_parameter_grid format
        Example: {"gemm-dtype": ["auto", "fp8"], "kv-cache-dtype": ["auto", "fp8_e5m2"]}

    Note:
        - Single values are wrapped in lists
        - Keys are converted to CLI flag format (underscore -> hyphen)
        - 'auto' values are filtered out (they mean "use default")
    """
    if not quant_config:
        return {}

    # Fields that map to runtime parameters
    dtype_fields = {
        "gemm_dtype": "gemm-dtype",
        "kvcache_dtype": "kv-cache-dtype",
        "attention_dtype": "attention-dtype",
        "moe_dtype": "moe-dtype"
    }

    param_spec = {}

    for field, cli_name in dtype_fields.items():
        if field not in quant_config:
            continue

        value = quant_config[field]

        # Convert to list if single value
        if not isinstance(value, list):
            value = [value]

        # Filter out 'auto' as it means "use default" (no CLI flag needed)
        # Keep other values for parameter grid
        filtered_values = [v for v in value if v != "auto"]

        # Only add to spec if there are non-auto values
        if filtered_values:
            param_spec[cli_name] = filtered_values

    return param_spec


def merge_parameters_with_quant_config(
    base_parameters: Dict[str, Any],
    quant_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge base parameters with quant_config parameter spec.

    This combines user-defined parameters with quantization parameters,
    preparing a unified parameter specification for grid expansion.

    Args:
        base_parameters: User-defined parameters (e.g., {"tp-size": [1, 2]})
        quant_config: Quantization configuration with arrays

    Returns:
        Merged parameter specification

    Example:
        >>> merge_parameters_with_quant_config(
        ...     {"tp-size": [1, 2]},
        ...     {"gemm_dtype": ["auto", "fp8"], "kvcache_dtype": ["fp8_e5m2"]}
        ... )
        {"tp-size": [1, 2], "gemm-dtype": ["fp8"], "kv-cache-dtype": ["fp8_e5m2"]}
    """
    # Start with base parameters
    merged = base_parameters.copy() if base_parameters else {}

    # Add quant_config parameters
    quant_params = expand_quant_config_to_parameter_spec(quant_config)

    # Merge (quant_params can override base_parameters if same key exists)
    merged.update(quant_params)

    return merged
