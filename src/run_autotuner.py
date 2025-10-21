#!/usr/bin/env python3
"""
Inference Autotuner - Main Orchestrator

Prototype implementation that reads a JSON task file and orchestrates
OME deployments and genai-bench benchmarks to find optimal parameters.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.ome_controller import OMEController
from controllers.benchmark_controller import BenchmarkController
from utils.optimizer import generate_parameter_grid, calculate_objective_score


class AutotunerOrchestrator:
    """Main orchestrator for the autotuning process."""

    def __init__(self, kubeconfig_path: str = None):
        """Initialize the orchestrator.

        Args:
            kubeconfig_path: Path to kubeconfig file
        """
        self.ome_controller = OMEController(kubeconfig_path)
        self.benchmark_controller = BenchmarkController(kubeconfig_path)
        self.results = []

    def load_task(self, task_file: Path) -> Dict[str, Any]:
        """Load task configuration from JSON file.

        Args:
            task_file: Path to task JSON file

        Returns:
            Task configuration dictionary
        """
        with open(task_file) as f:
            task = json.load(f)
        print(f"Loaded task: {task['task_name']}")
        print(f"Description: {task['description']}")
        return task

    def run_experiment(
        self,
        task: Dict[str, Any],
        experiment_id: int,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        model_name = task["model"]["name"]
        runtime_name = task["base_runtime"]
        timeout = task["optimization"]["timeout_per_iteration"]

        print(f"\n{'='*80}")
        print(f"Experiment {experiment_id}")
        print(f"Parameters: {parameters}")
        print(f"{'='*80}\n")

        experiment_result = {
            "experiment_id": experiment_id,
            "parameters": parameters,
            "status": "failed",
            "metrics": None
        }

        # Step 1: Deploy InferenceService
        print(f"[Step 1/4] Deploying InferenceService...")
        isvc_name = self.ome_controller.deploy_inference_service(
            task_name=task_name,
            experiment_id=experiment_id,
            namespace=namespace,
            model_name=model_name,
            runtime_name=runtime_name,
            parameters=parameters
        )

        if not isvc_name:
            print("Failed to deploy InferenceService")
            return experiment_result

        # Step 2: Wait for InferenceService to be ready
        print(f"\n[Step 2/4] Waiting for InferenceService to be ready...")
        if not self.ome_controller.wait_for_ready(isvc_name, namespace, timeout=timeout):
            print("InferenceService did not become ready in time")
            self.cleanup_experiment(isvc_name, None, namespace)
            return experiment_result

        # Step 3: Run benchmark
        print(f"\n[Step 3/4] Running benchmark...")
        benchmark_name = self.benchmark_controller.create_benchmark_job(
            task_name=task_name,
            experiment_id=experiment_id,
            namespace=namespace,
            isvc_name=isvc_name,
            benchmark_config=task["benchmark"]
        )

        if not benchmark_name:
            print("Failed to create BenchmarkJob")
            self.cleanup_experiment(isvc_name, None, namespace)
            return experiment_result

        # Wait for benchmark to complete
        if not self.benchmark_controller.wait_for_completion(
            benchmark_name, namespace, timeout=timeout
        ):
            print("Benchmark did not complete in time")
            self.cleanup_experiment(isvc_name, benchmark_name, namespace)
            return experiment_result

        # Step 4: Collect results
        print(f"\n[Step 4/4] Collecting results...")
        metrics = self.benchmark_controller.get_benchmark_results(
            benchmark_name, namespace
        )

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
        self.cleanup_experiment(isvc_name, benchmark_name, namespace)

        return experiment_result

    def cleanup_experiment(
        self,
        isvc_name: str,
        benchmark_name: str,
        namespace: str
    ):
        """Clean up experiment resources.

        Args:
            isvc_name: InferenceService name
            benchmark_name: BenchmarkJob name (can be None)
            namespace: K8s namespace
        """
        if benchmark_name:
            self.benchmark_controller.delete_benchmark_job(benchmark_name, namespace)
        if isvc_name:
            self.ome_controller.delete_inference_service(isvc_name, namespace)

    def run_task(self, task_file: Path) -> Dict[str, Any]:
        """Run a complete autotuning task.

        Args:
            task_file: Path to task JSON file

        Returns:
            Summary of all experiments
        """
        # Load task
        task = self.load_task(task_file)

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
            print("\nNo successful experiments!")

        return {
            "task_name": task["task_name"],
            "total_experiments": len(self.results),
            "successful_experiments": len(successful_results),
            "elapsed_time": elapsed,
            "best_result": best_result if successful_results else None,
            "all_results": self.results
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_autotuner.py <task_json_file> [kubeconfig_path]")
        sys.exit(1)

    task_file = Path(sys.argv[1])
    if not task_file.exists():
        print(f"Error: Task file not found: {task_file}")
        sys.exit(1)

    kubeconfig_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Run autotuning
    orchestrator = AutotunerOrchestrator(kubeconfig_path)
    summary = orchestrator.run_task(task_file)

    # Save results
    results_file = Path("results") / f"{summary['task_name']}_results.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
