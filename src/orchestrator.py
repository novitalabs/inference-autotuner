"""
Autotuner Orchestrator

Main orchestration logic for running parameter tuning experiments.
Coordinates deployment controllers and benchmark execution.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from controllers.ome_controller import OMEController
from controllers.docker_controller import DockerController
from controllers.benchmark_controller import BenchmarkController
from controllers.direct_benchmark_controller import DirectBenchmarkController
from utils.optimizer import generate_parameter_grid, calculate_objective_score


class AutotunerOrchestrator:
	"""Main orchestrator for the autotuning process."""

	def __init__(
		self,
		deployment_mode: str = "ome",
		kubeconfig_path: str = None,
		use_direct_benchmark: bool = False,
		docker_model_path: str = "/mnt/data/models",
		verbose: bool = False,
		http_proxy: str = "",
		https_proxy: str = "",
		no_proxy: str = "",
		hf_token: str = "",
	):
		"""Initialize the orchestrator.

		Args:
		    deployment_mode: Deployment mode - 'ome' (Kubernetes) or 'docker' (standalone)
		    kubeconfig_path: Path to kubeconfig file (for OME mode)
		    use_direct_benchmark: If True, use direct genai-bench CLI instead of K8s BenchmarkJob
		    docker_model_path: Base path for models in Docker mode
		    verbose: If True, stream genai-bench output in real-time
		    http_proxy: HTTP proxy URL for containers (optional)
		    https_proxy: HTTPS proxy URL for containers (optional)
		    no_proxy: Comma-separated list of hosts to bypass proxy (optional)
		"""
		self.deployment_mode = deployment_mode.lower()
		self.use_direct_benchmark = use_direct_benchmark

		# Initialize model deployment controller based on mode
		if self.deployment_mode == "docker":
			print("[Config] Deployment mode: Standalone Docker")
			self.model_controller = DockerController(
				model_base_path=docker_model_path,
				http_proxy=http_proxy,
				https_proxy=https_proxy,
				no_proxy=no_proxy,
				hf_token=hf_token
			)
			# Docker mode always uses direct benchmark (no K8s)
			self.use_direct_benchmark = True
			self.benchmark_controller = DirectBenchmarkController(verbose=verbose)
			print("[Config] Benchmark mode: Direct CLI (automatic for Docker mode)")
			print("[Config] Containers will be auto-removed after stop")
		elif self.deployment_mode == "ome":
			print("[Config] Deployment mode: OME (Kubernetes)")
			self.model_controller = OMEController(kubeconfig_path)
			if use_direct_benchmark:
				self.benchmark_controller = DirectBenchmarkController(verbose=verbose)
				print("[Config] Benchmark mode: Direct genai-bench CLI")
			else:
				self.benchmark_controller = BenchmarkController(kubeconfig_path)
				print("[Config] Benchmark mode: Kubernetes BenchmarkJob CRD")
		else:
			raise ValueError(f"Unknown deployment mode: {deployment_mode}. Use 'ome' or 'docker'")

		self.results = []

	def run_experiment(self, task: Dict[str, Any], experiment_id: int, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Run a single tuning experiment.

		Args:
		    task: Task configuration
		    experiment_id: Unique experiment identifier
		    parameters: Parameter values for this experiment

		Returns:
		    Experiment results dictionary
		"""
		task_name = task["task_name"]
		namespace = task["model"]["namespace"]
		model_name = task["model"]["id_or_path"]
		runtime_name = task["base_runtime"]
		timeout = task["optimization"]["timeout_per_iteration"]

		# Optional: custom Docker image tag
		image_tag = task.get("runtime_image_tag")

		print(f"\n{'='*80}")
		print(f"Experiment {experiment_id}")
		print(f"Parameters: {parameters}")
		print(f"{'='*80}\n")

		experiment_result = {
			"experiment_id": experiment_id,
			"parameters": parameters,
			"status": "failed",
			"metrics": None,
			"container_logs": None,  # Will store container logs for Docker mode
		}

		# Step 1: Deploy InferenceService
		print(f"[Step 1/4] Deploying InferenceService...")

		# Pass image_tag if deploying with DockerController
		if hasattr(self.model_controller, "client"):  # DockerController has 'client' attribute
			isvc_name = self.model_controller.deploy_inference_service(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				model_name=model_name,
				runtime_name=runtime_name,
				parameters=parameters,
				image_tag=image_tag,
			)
		else:  # OMEController doesn't support image_tag yet
			isvc_name = self.model_controller.deploy_inference_service(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				model_name=model_name,
				runtime_name=runtime_name,
				parameters=parameters,
			)

		if not isvc_name:
			print("Failed to deploy InferenceService")
			return experiment_result

		# Step 2: Wait for InferenceService to be ready
		print(f"\n[Step 2/4] Waiting for InferenceService to be ready...")
		if not self.model_controller.wait_for_ready(isvc_name, namespace, timeout=timeout):
			print("InferenceService did not become ready in time")
			self.cleanup_experiment(isvc_name, None, namespace)
			return experiment_result

		# Step 3: Run benchmark
		print(f"\n[Step 3/4] Running benchmark...")

		if self.use_direct_benchmark:
			# Get endpoint URL (differs between Docker and OME modes)
			endpoint_url = None
			if self.deployment_mode == "docker":
				# Docker mode: Get direct URL from controller
				endpoint_url = self.model_controller.get_service_url(isvc_name, namespace)
				if not endpoint_url:
					print("Failed to get service URL from Docker controller")
					self.cleanup_experiment(isvc_name, None, namespace, experiment_id)
					return experiment_result

			# Direct CLI execution with automatic port forwarding (OME) or direct URL (Docker)
			metrics = self.benchmark_controller.run_benchmark(
				task_name=task_name,
				experiment_id=experiment_id,
				service_name=isvc_name,
				namespace=namespace,
				benchmark_config=task["benchmark"],
				timeout=timeout,
				endpoint_url=endpoint_url,
			)
			benchmark_name = None  # No K8s resource to track

			# Step 4: Process results
			print(f"\n[Step 4/4] Processing results...")
			if metrics:
				experiment_result["status"] = "success"
				experiment_result["metrics"] = metrics
				score = calculate_objective_score(metrics, task["optimization"]["objective"])
				experiment_result["objective_score"] = score
				print(f"Experiment {experiment_id} completed. Score: {score}")
			else:
				print("Failed to retrieve benchmark results")
		else:
			# K8s BenchmarkJob execution
			benchmark_name = self.benchmark_controller.create_benchmark_job(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				isvc_name=isvc_name,
				benchmark_config=task["benchmark"],
			)

			if not benchmark_name:
				print("Failed to create BenchmarkJob")
				self.cleanup_experiment(isvc_name, None, namespace)
				return experiment_result

			# Wait for benchmark to complete
			if not self.benchmark_controller.wait_for_completion(benchmark_name, namespace, timeout=timeout):
				print("Benchmark did not complete in time")
				self.cleanup_experiment(isvc_name, benchmark_name, namespace)
				return experiment_result

			# Step 4: Collect results
			print(f"\n[Step 4/4] Collecting results...")
			metrics = self.benchmark_controller.get_benchmark_results(benchmark_name, namespace)

			if metrics:
				experiment_result["status"] = "success"
				experiment_result["metrics"] = metrics
				score = calculate_objective_score(metrics, task["optimization"]["objective"])
				experiment_result["objective_score"] = score
				print(f"Experiment {experiment_id} completed. Score: {score}")
			else:
				print("Failed to retrieve benchmark results")

		# Cleanup
		print(f"\n[Cleanup] Removing experiment resources...")
		container_logs = self.cleanup_experiment(isvc_name, benchmark_name, namespace, experiment_id)

		# Store container logs in result if available
		if container_logs:
			experiment_result["container_logs"] = container_logs

		return experiment_result

	def cleanup_experiment(self, isvc_name: str, benchmark_name: str, namespace: str, experiment_id: int = None) -> str:
		"""Clean up experiment resources.

		Args:
		    isvc_name: InferenceService name
		    benchmark_name: BenchmarkJob name (can be None)
		    namespace: K8s namespace
		    experiment_id: Experiment ID (for direct benchmark cleanup)

		Returns:
		    Container logs if available (Docker mode only), None otherwise
		"""
		container_logs = None

		# Retrieve container logs before deletion (Docker mode only)
		if self.deployment_mode == "docker" and hasattr(self.model_controller, "get_container_logs"):
			print(f"[Cleanup] Retrieving container logs before deletion...")
			container_logs = self.model_controller.get_container_logs(isvc_name, namespace, tail=0)  # Get ALL logs
			if container_logs:
				print(f"[Cleanup] Retrieved {len(container_logs)} bytes of container logs")

		# Continue with cleanup
		if self.use_direct_benchmark and experiment_id:
			self.benchmark_controller.cleanup_results(
				task_name=isvc_name.rsplit("-exp", 1)[0], experiment_id=experiment_id
			)
		elif benchmark_name:
			self.benchmark_controller.delete_benchmark_job(benchmark_name, namespace)

		if isvc_name:
			self.model_controller.delete_inference_service(isvc_name, namespace)

		return container_logs

	def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
		"""Run a complete autotuning task.

		Args:
		    task: Task configuration dictionary

		Returns:
		    Summary of all experiments
		"""
		task_name = task["task_name"]
		print(f"Starting task: {task_name}")
		if "description" in task:
			print(f"Description: {task['description']}")

		# Generate parameter grid
		param_grid = generate_parameter_grid(task["parameters"])
		print(f"\nGenerated parameter grid with {len(param_grid)} combinations:")
		for i, params in enumerate(param_grid):
			print(f"  {i+1}. {params}")

		# Limit by max_iterations
		max_iterations = task["optimization"].get("max_iterations", len(param_grid))
		param_grid = param_grid[:max_iterations]
		print(f"\nWill run {len(param_grid)} experiments (limited by max_iterations)")

		# Run experiments
		start_time = time.time()
		for i, parameters in enumerate(param_grid, start=1):
			result = self.run_experiment(task, i, parameters)
			self.results.append(result)

		elapsed = time.time() - start_time

		# Find best result
		successful_results = [r for r in self.results if r["status"] == "success"]
		if successful_results:
			best_result = min(successful_results, key=lambda r: r["objective_score"])
			print(f"\n{'='*80}")
			print(f"AUTOTUNING COMPLETE")
			print(f"{'='*80}")
			print(f"Total experiments: {len(self.results)}")
			print(f"Successful: {len(successful_results)}")
			print(f"Failed: {len(self.results) - len(successful_results)}")
			print(f"Total time: {elapsed:.1f}s")
			print(f"\nBest configuration:")
			print(f"  Parameters: {best_result['parameters']}")
			print(f"  Score: {best_result['objective_score']}")
		else:
			best_result = None
			print("\nNo successful experiments!")

		return {
			"task_name": task_name,
			"total_experiments": len(self.results),
			"successful_experiments": len(successful_results),
			"elapsed_time": elapsed,
			"best_result": best_result,
			"all_results": self.results,
		}
