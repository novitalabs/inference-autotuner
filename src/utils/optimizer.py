"""
Utility functions for parameter optimization.
"""

import itertools
from typing import Dict, List, Any


def generate_parameter_grid(parameter_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for grid search.

    Supports two formats:
    1. Simple format: {"param_name": [value1, value2]}
       - Direct list of values for each parameter
    2. Structured format: {"param_name": {"type": "choice", "values": [value1, value2]}}
       - Legacy format with explicit type specification

    Args:
        parameter_spec: Dict mapping parameter names to their values or specifications
                       Simple: {"tp-size": [1, 2], "mem-fraction-static": [0.7, 0.8]}
                       Structured: {"tp_size": {"type": "choice", "values": [1, 2]}}

    Returns:
        List of parameter dictionaries, one for each combination
    """
    param_names = []
    param_values = []

    for param_name, spec in parameter_spec.items():
        # Check if it's the simple format (direct list) or structured format (dict with type/values)
        if isinstance(spec, list):
            # Simple format: direct list of values
            param_names.append(param_name)
            param_values.append(spec)
        elif isinstance(spec, dict) and "type" in spec:
            # Structured format: legacy support
            if spec["type"] == "choice":
                param_names.append(param_name)
                param_values.append(spec["values"])
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        else:
            raise ValueError(
                f"Invalid parameter specification for '{param_name}'. "
                f"Expected list of values or dict with 'type' and 'values' keys."
            )

    # Generate Cartesian product
    combinations = list(itertools.product(*param_values))

    # Convert to list of dicts
    grid = []
    for combo in combinations:
        param_dict = dict(zip(param_names, combo))
        grid.append(param_dict)

    return grid


def calculate_objective_score(
    results: Dict[str, Any],
    objective: str = "minimize_latency"
) -> float:
    """Calculate objective score from benchmark results.

    Args:
        results: Benchmark results dictionary
        objective: Optimization objective

    Returns:
        Objective score (lower is better for minimization)
    """
    # This is a placeholder - actual implementation depends on
    # the structure of results returned by BenchmarkJob

    if objective == "minimize_latency":
        # Extract average latency (example)
        # Actual field names depend on genai-bench output format
        latency = results.get("results", {}).get("avg_e2e_latency", float('inf'))
        return latency
    elif objective == "maximize_throughput":
        throughput = results.get("results", {}).get("throughput", 0)
        return -throughput  # Negate for minimization
    else:
        raise ValueError(f"Unsupported objective: {objective}")
