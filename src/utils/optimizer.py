"""
Utility functions and classes for parameter optimization.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import optuna


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
			# Skip empty lists (no values to search)
			if len(spec) == 0:
				continue
			param_names.append(param_name)
			param_values.append(spec)
		elif isinstance(spec, dict) and "type" in spec:
			# Structured format: legacy support
			if spec["type"] == "choice":
				# Skip empty value lists
				values = spec["values"]
				if len(values) == 0:
					continue
				param_names.append(param_name)
				param_values.append(values)
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


# ============================================================================
# Optimization Strategy Abstraction
# ============================================================================


class OptimizationStrategy(ABC):
	"""Abstract base class for optimization strategies."""

	def __init__(self, parameter_spec: Dict[str, Any], objective: str = "minimize_latency"):
		"""Initialize optimization strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective (minimize_latency, maximize_throughput, etc.)
		"""
		self.parameter_spec = parameter_spec
		self.objective = objective
		self.history: List[Dict[str, Any]] = []

	@abstractmethod
	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest next parameter configuration to try.

		Returns:
		    Dictionary of parameter values, or None if strategy is done
		"""
		pass

	@abstractmethod
	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Update strategy with experiment result.

		Args:
		    parameters: Parameter configuration that was tested
		    objective_score: Objective score from calculate_objective_score()
		    metrics: Full metrics dictionary from benchmark
		"""
		pass

	def should_stop(self) -> bool:
		"""Check if optimization should stop early.

		Returns:
		    True if strategy has converged or no more suggestions
		"""
		return False


class GridSearchStrategy(OptimizationStrategy):
	"""Grid search optimization - exhaustive evaluation of all combinations."""

	def __init__(self, parameter_spec: Dict[str, Any], objective: str = "minimize_latency", max_iterations: Optional[int] = None):
		"""Initialize grid search strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of iterations (limits grid size)
		"""
		super().__init__(parameter_spec, objective)
		self.param_grid = generate_parameter_grid(parameter_spec)

		# Limit grid size if max_iterations specified
		if max_iterations is not None and max_iterations < len(self.param_grid):
			print(f"[GridSearch] Limiting grid from {len(self.param_grid)} to {max_iterations} combinations")
			self.param_grid = self.param_grid[:max_iterations]

		self.current_index = 0
		print(f"[GridSearch] Initialized with {len(self.param_grid)} parameter combinations")

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Get next parameter combination from grid."""
		if self.current_index >= len(self.param_grid):
			return None

		params = self.param_grid[self.current_index]
		self.current_index += 1
		print(f"[GridSearch] Suggesting combination {self.current_index}/{len(self.param_grid)}: {params}")
		return params

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Record result (grid search doesn't adapt)."""
		self.history.append({
			"parameters": parameters,
			"objective_score": objective_score,
			"metrics": metrics
		})
		print(f"[GridSearch] Recorded result: score={objective_score:.4f}")

	def should_stop(self) -> bool:
		"""Stop when all combinations evaluated."""
		return self.current_index >= len(self.param_grid)


class BayesianStrategy(OptimizationStrategy):
	"""Bayesian optimization using Optuna's TPE sampler."""

	def __init__(
		self,
		parameter_spec: Dict[str, Any],
		objective: str = "minimize_latency",
		max_iterations: int = 100,
		n_initial_random: int = 5,
		study_name: Optional[str] = None,
		storage: Optional[str] = None
	):
		"""Initialize Bayesian optimization strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of trials
		    n_initial_random: Number of random trials before Bayesian optimization
		    study_name: Optional name for Optuna study
		    storage: Optional Optuna storage URL (e.g., sqlite:///optuna.db)
		"""
		super().__init__(parameter_spec, objective)
		self.max_iterations = max_iterations
		self.n_initial_random = n_initial_random
		self.trial_count = 0

		# Parse parameter specification into Optuna search space
		self.search_space = self._parse_search_space(parameter_spec)

		# Create Optuna study
		direction = "minimize"  # All objectives use minimize (throughput is negated)
		sampler = optuna.samplers.TPESampler(n_startup_trials=n_initial_random)

		self.study = optuna.create_study(
			direction=direction,
			sampler=sampler,
			study_name=study_name or f"autotuner_{objective}",
			storage=storage,
			load_if_exists=True  # Resume if study exists
		)

		print(f"[Bayesian] Initialized with {len(self.search_space)} parameters")
		print(f"[Bayesian] Max iterations: {max_iterations}, Initial random: {n_initial_random}")
		print(f"[Bayesian] Search space: {list(self.search_space.keys())}")

	def _parse_search_space(self, parameter_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		"""Parse parameter specification into Optuna search space.

		Supports:
		- Simple format: {"param": [val1, val2]} → categorical
		- Explicit format: {"param": {"type": "categorical", "values": [...]}}
		- Continuous: {"param": {"type": "continuous", "low": 0.5, "high": 1.0}}
		- Integer: {"param": {"type": "integer", "low": 1, "high": 10}}

		Returns:
		    Dict mapping parameter names to Optuna space definitions
		"""
		search_space = {}

		for param_name, spec in parameter_spec.items():
			if isinstance(spec, list):
				# Simple format: treat as categorical
				# Skip empty lists (no values to search)
				if len(spec) == 0:
					continue
				search_space[param_name] = {
					"type": "categorical",
					"choices": spec
				}
			elif isinstance(spec, dict):
				param_type = spec.get("type", "categorical")

				if param_type in ["choice", "categorical"]:
					# Categorical parameter
					values = spec.get("values", spec.get("choices", []))
					# Skip empty value lists
					if len(values) == 0:
						continue
					search_space[param_name] = {
						"type": "categorical",
						"choices": values
					}
				elif param_type == "continuous":
					# Continuous (float) parameter
					search_space[param_name] = {
						"type": "continuous",
						"low": spec["low"],
						"high": spec["high"],
						"log": spec.get("log", False)  # Log scale for parameters like learning rate
					}
				elif param_type == "integer":
					# Integer parameter
					search_space[param_name] = {
						"type": "integer",
						"low": spec["low"],
						"high": spec["high"],
						"log": spec.get("log", False)
					}
				else:
					raise ValueError(f"Unsupported parameter type: {param_type}")
			else:
				raise ValueError(f"Invalid parameter specification for '{param_name}'")

		return search_space

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest next parameter configuration using Optuna."""
		if self.trial_count >= self.max_iterations:
			print(f"[Bayesian] Reached max iterations ({self.max_iterations})")
			return None

		# Create Optuna trial
		trial = self.study.ask()
		self.current_trial = trial
		self.trial_count += 1

		# Extract parameter suggestions
		params = {}
		for param_name, space_def in self.search_space.items():
			if space_def["type"] == "categorical":
				params[param_name] = trial.suggest_categorical(param_name, space_def["choices"])
			elif space_def["type"] == "continuous":
				params[param_name] = trial.suggest_float(
					param_name,
					space_def["low"],
					space_def["high"],
					log=space_def.get("log", False)
				)
			elif space_def["type"] == "integer":
				params[param_name] = trial.suggest_int(
					param_name,
					space_def["low"],
					space_def["high"],
					log=space_def.get("log", False)
				)

		print(f"[Bayesian] Trial {self.trial_count}/{self.max_iterations}: {params}")
		return params

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Update Optuna study with experiment result."""
		# Tell Optuna the result
		self.study.tell(self.current_trial, objective_score)

		# Record in history
		self.history.append({
			"parameters": parameters,
			"objective_score": objective_score,
			"metrics": metrics
		})

		# Print progress
		best_score = self.study.best_value
		best_params = self.study.best_params
		print(f"[Bayesian] Trial complete: score={objective_score:.4f}")
		print(f"[Bayesian] Best so far: score={best_score:.4f}, params={best_params}")

	def should_stop(self) -> bool:
		"""Check if Bayesian optimization should stop."""
		# Stop if max iterations reached
		if self.trial_count >= self.max_iterations:
			return True

		# Optional: Add convergence detection
		# e.g., no improvement in last N trials
		# For now, just use max_iterations

		return False

	def get_best_params(self) -> Dict[str, Any]:
		"""Get best parameters found so far."""
		return self.study.best_params

	def get_best_score(self) -> float:
		"""Get best objective score found so far."""
		return self.study.best_value


class RandomSearchStrategy(OptimizationStrategy):
	"""Random search - random sampling from parameter space."""

	def __init__(
		self,
		parameter_spec: Dict[str, Any],
		objective: str = "minimize_latency",
		max_iterations: int = 100,
		seed: Optional[int] = None
	):
		"""Initialize random search strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of random samples
		    seed: Random seed for reproducibility
		"""
		super().__init__(parameter_spec, objective)
		self.max_iterations = max_iterations
		self.trial_count = 0

		# Use Optuna's RandomSampler for convenience
		import random
		if seed is not None:
			random.seed(seed)

		sampler = optuna.samplers.RandomSampler(seed=seed)
		self.study = optuna.create_study(
			direction="minimize",
			sampler=sampler,
			study_name=f"autotuner_random_{objective}"
		)

		# Parse search space (reuse BayesianStrategy's parser)
		bayesian_helper = BayesianStrategy(parameter_spec, objective, max_iterations=1)
		self.search_space = bayesian_helper.search_space

		print(f"[Random] Initialized with {max_iterations} random samples")

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest random parameter configuration."""
		if self.trial_count >= self.max_iterations:
			return None

		# Create Optuna trial with random sampler
		trial = self.study.ask()
		self.current_trial = trial
		self.trial_count += 1

		# Extract parameters
		params = {}
		for param_name, space_def in self.search_space.items():
			if space_def["type"] == "categorical":
				params[param_name] = trial.suggest_categorical(param_name, space_def["choices"])
			elif space_def["type"] == "continuous":
				params[param_name] = trial.suggest_float(
					param_name,
					space_def["low"],
					space_def["high"],
					log=space_def.get("log", False)
				)
			elif space_def["type"] == "integer":
				params[param_name] = trial.suggest_int(
					param_name,
					space_def["low"],
					space_def["high"],
					log=space_def.get("log", False)
				)

		print(f"[Random] Sample {self.trial_count}/{self.max_iterations}: {params}")
		return params

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Record result."""
		self.study.tell(self.current_trial, objective_score)
		self.history.append({
			"parameters": parameters,
			"objective_score": objective_score,
			"metrics": metrics
		})

	def should_stop(self) -> bool:
		"""Stop after max iterations."""
		return self.trial_count >= self.max_iterations


# ============================================================================
# Strategy Factory
# ============================================================================


def create_optimization_strategy(
	optimization_config: Dict[str, Any],
	parameter_spec: Dict[str, Any]
) -> OptimizationStrategy:
	"""Factory function to create optimization strategy.

	Args:
	    optimization_config: Optimization configuration from task
	                        {"strategy": "grid_search", "objective": "minimize_latency", ...}
	    parameter_spec: Parameter specification dictionary

	Returns:
	    OptimizationStrategy instance
	"""
	strategy_name = optimization_config.get("strategy", "grid_search")
	objective = optimization_config.get("objective", "minimize_latency")
	max_iterations = optimization_config.get("max_iterations", 100)

	if strategy_name == "grid_search":
		return GridSearchStrategy(
			parameter_spec=parameter_spec,
			objective=objective,
			max_iterations=max_iterations
		)

	elif strategy_name == "bayesian":
		n_initial_random = optimization_config.get("n_initial_random", 5)
		study_name = optimization_config.get("study_name")
		storage = optimization_config.get("storage")  # e.g., "sqlite:///optuna.db"

		return BayesianStrategy(
			parameter_spec=parameter_spec,
			objective=objective,
			max_iterations=max_iterations,
			n_initial_random=n_initial_random,
			study_name=study_name,
			storage=storage
		)

	elif strategy_name == "random":
		seed = optimization_config.get("seed")

		return RandomSearchStrategy(
			parameter_spec=parameter_spec,
			objective=objective,
			max_iterations=max_iterations,
			seed=seed
		)

	else:
		raise ValueError(
			f"Unsupported optimization strategy: {strategy_name}. "
			f"Supported: grid_search, bayesian, random"
		)
