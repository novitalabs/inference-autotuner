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


def calculate_objective_score(results: Dict[str, Any], objective: str = "minimize_latency") -> float:
	"""Calculate objective score from benchmark results.

	Args:
	    results: Benchmark results dictionary from DirectBenchmarkController._parse_results()
	    objective: Optimization objective - 'minimize_latency' or 'maximize_throughput'

	Returns:
	    Objective score (lower is better for minimization, higher for maximization)
	"""
	if not results:
		print("[Optimizer] No results provided, returning worst score")
		return float("inf") if "minimize" in objective else float("-inf")

	# Extract metrics based on objective
	try:
		if objective == "minimize_latency":
			# Use mean E2E latency as primary metric (in seconds)
			# Fallback to P50 if mean not available
			latency = results.get("mean_e2e_latency", results.get("p50_e2e_latency"))
			if latency is None:
				print(f"[Optimizer] Warning: No latency metrics found in results")
				print(f"[Optimizer] Available keys: {list(results.keys())}")
				return float("inf")

			print(f"[Optimizer] Latency score: {latency:.4f}s (lower is better)")
			return latency

		elif objective == "maximize_throughput":
			# Use mean total throughput (tokens/s) as primary metric
			# Fallback to output throughput or max throughput
			throughput = results.get(
				"mean_total_throughput", results.get("mean_output_throughput", results.get("max_total_throughput"))
			)
			if throughput is None or throughput == 0:
				print(f"[Optimizer] Warning: No throughput metrics found in results")
				print(f"[Optimizer] Available keys: {list(results.keys())}")
				return float("-inf")

			# Negate for minimization (optimizer looks for minimum score)
			score = -throughput
			print(f"[Optimizer] Throughput score: {throughput:.2f} tokens/s (score: {score:.2f}, lower is better)")
			return score

		elif objective == "minimize_ttft":
			# Time to First Token optimization
			ttft = results.get("mean_ttft")
			if ttft is None:
				print(f"[Optimizer] Warning: No TTFT metrics found in results")
				return float("inf")

			print(f"[Optimizer] TTFT score: {ttft:.4f}s (lower is better)")
			return ttft

		elif objective == "minimize_tpot":
			# Time Per Output Token optimization
			tpot = results.get("mean_tpot")
			if tpot is None:
				print(f"[Optimizer] Warning: No TPOT metrics found in results")
				return float("inf")

			print(f"[Optimizer] TPOT score: {tpot:.4f}s (lower is better)")
			return tpot

		else:
			raise ValueError(f"Unsupported objective: {objective}")

	except Exception as e:
		print(f"[Optimizer] Error calculating objective score: {e}")
		return float("inf") if "minimize" in objective else float("-inf")
