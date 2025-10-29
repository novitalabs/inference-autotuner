"""
Direct GenAI-Bench Controller

Runs genai-bench directly using the CLI instead of Kubernetes BenchmarkJob CRDs.
This bypasses the genai-bench v251014 image issues by using the local installation.
"""

import json
import subprocess
import time
import signal
from pathlib import Path
from typing import Dict, Any, Optional


class DirectBenchmarkController:
	"""Controller for running genai-bench directly via CLI."""

	def __init__(self, genai_bench_path: str = "env/bin/genai-bench", verbose: bool = False):
		"""Initialize the direct benchmark controller.

		Args:
		    genai_bench_path: Path to genai-bench executable (can be relative or absolute)
		    verbose: If True, stream genai-bench output in real-time
		"""
		# Convert to Path and resolve to absolute path
		genai_bench_path_obj = Path(genai_bench_path)

		# If relative path, resolve relative to project root
		if not genai_bench_path_obj.is_absolute():
			# Try to find project root (where src/ directory is located)
			current_file = Path(__file__).resolve()  # controllers/direct_benchmark_controller.py
			project_root = current_file.parent.parent.parent  # Go up to inference-autotuner/
			genai_bench_path_obj = project_root / genai_bench_path_obj

		self.genai_bench_path = genai_bench_path_obj
		if not self.genai_bench_path.exists():
			raise FileNotFoundError(f"genai-bench not found at {self.genai_bench_path}")

		self.verbose = verbose

		# Results directory - always resolve relative to project root
		current_file = Path(__file__).resolve()  # controllers/direct_benchmark_controller.py
		project_root = current_file.parent.parent.parent  # Go up to inference-autotuner/
		self.results_dir = project_root / "benchmark_results"
		self.results_dir.mkdir(exist_ok=True)

		# Port forward process tracking
		self.port_forward_proc = None
		self.local_port = None

	def setup_port_forward(
		self, service_name: str, namespace: str, remote_port: int = 8000, local_port: int = 8080
	) -> Optional[str]:
		"""Setup kubectl port-forward for accessing InferenceService.

		Args:
		    service_name: InferenceService name (used to find pods)
		    namespace: K8s namespace
		    remote_port: Remote service port (default 8000 for SGLang)
		    local_port: Local port to forward to

		Returns:
		    Local endpoint URL if successful, None otherwise
		"""
		print(f"[Port Forward] Setting up port forward for {service_name}.{namespace}...")

		# First, find the pod for this InferenceService
		# OME creates pods with labels matching the InferenceService
		try:
			import subprocess as sp

			result = sp.run(
				[
					"kubectl",
					"get",
					"pods",
					"-n",
					namespace,
					"-l",
					f"serving.kserve.io/inferenceservice={service_name}",
					"-o",
					"jsonpath={.items[0].metadata.name}",
				],
				capture_output=True,
				text=True,
				timeout=10,
			)

			if result.returncode != 0 or not result.stdout.strip():
				print(f"[Port Forward] No pods found for InferenceService {service_name}")
				print(f"[Port Forward] Trying direct service name...")
				# Fallback to service name
				pod_or_svc = f"svc/{service_name}"
			else:
				pod_name = result.stdout.strip()
				print(f"[Port Forward] Found pod: {pod_name}")
				pod_or_svc = f"pod/{pod_name}"

		except Exception as e:
			print(f"[Port Forward] Error finding pod: {e}")
			pod_or_svc = f"svc/{service_name}"

		# Start port-forward in background
		cmd = [
			"kubectl",
			"port-forward",
			pod_or_svc,
			f"{local_port}:{remote_port}",
			"-n",
			namespace,
		]

		try:
			self.port_forward_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
			self.local_port = local_port

			# Wait a bit for port forward to establish
			time.sleep(5)

			# Check if process is still running
			if self.port_forward_proc.poll() is not None:
				stderr = self.port_forward_proc.stderr.read()
				print(f"[Port Forward] Failed to establish: {stderr}")
				return None

			endpoint = f"http://localhost:{local_port}"
			print(f"[Port Forward] Established: {endpoint}")
			return endpoint

		except Exception as e:
			print(f"[Port Forward] Error: {e}")
			return None

	def cleanup_port_forward(self):
		"""Stop port forward process."""
		if self.port_forward_proc:
			print(f"[Port Forward] Stopping port forward on localhost:{self.local_port}")
			try:
				self.port_forward_proc.send_signal(signal.SIGTERM)
				self.port_forward_proc.wait(timeout=5)
			except subprocess.TimeoutExpired:
				self.port_forward_proc.kill()
			except Exception as e:
				print(f"[Port Forward] Error stopping: {e}")
			finally:
				self.port_forward_proc = None
				self.local_port = None

	def run_benchmark(
		self,
		task_name: str,
		experiment_id: int,
		service_name: str,
		namespace: str,
		benchmark_config: Dict[str, Any],
		timeout: int = 1800,
		local_port: int = 8080,
		endpoint_url: Optional[str] = None,
	) -> Optional[Dict[str, Any]]:
		"""Run benchmark against an inference endpoint with automatic port forwarding.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    service_name: K8s service name (or Docker container name)
		    namespace: K8s namespace (ignored in Docker mode)
		    benchmark_config: Benchmark configuration from input JSON
		    timeout: Maximum execution time in seconds
		    local_port: Local port for port forwarding (ignored if endpoint_url is provided)
		    endpoint_url: Optional direct endpoint URL (skips port-forward setup for Docker mode)

		Returns:
		    Dict containing benchmark metrics, or None if failed
		"""
		benchmark_name = f"{task_name}-exp{experiment_id}"
		output_dir = self.results_dir / benchmark_name

		# Setup endpoint URL
		if endpoint_url:
			# Direct URL provided (Docker mode)
			print(f"[Benchmark] Using direct endpoint: {endpoint_url}")
			need_cleanup = False
		else:
			# Setup port forward (Kubernetes mode)
			endpoint_url = self.setup_port_forward(service_name, namespace, 8000, local_port)
			if not endpoint_url:
				print(f"[Benchmark] Failed to setup port forward")
				return None
			need_cleanup = True

		print(f"[Benchmark] Running genai-bench for experiment {experiment_id}")
		print(f"[Benchmark] Endpoint: {endpoint_url}")
		print(f"[Benchmark] Output directory: {output_dir}")

		# Build genai-bench command
		cmd = [
			str(self.genai_bench_path),
			"benchmark",
			"--api-backend",
			"openai",
			"--api-base",
			endpoint_url,
			"--api-key",
			"dummy",  # Required but not used for local servers
			"--task",
			benchmark_config.get("task", "text-to-text"),
			"--experiment-base-dir",
			str(output_dir.parent),
			"--experiment-folder-name",
			output_dir.name,
		]

		# Add model name (required by genai-bench)
		model_name = benchmark_config.get("model_name", "unknown")
		cmd.extend(["--api-model-name", model_name])

		# Add model tokenizer (required by genai-bench)
		model_tokenizer = benchmark_config.get("model_tokenizer", "gpt2")
		cmd.extend(["--model-tokenizer", model_tokenizer])

		# Add traffic scenarios
		traffic_scenarios = benchmark_config.get("traffic_scenarios", ["D(100,100)"])
		for scenario in traffic_scenarios:
			cmd.extend(["--traffic-scenario", scenario])

		# Add concurrency levels
		num_concurrency = benchmark_config.get("num_concurrency", [1])
		for concurrency in num_concurrency:
			cmd.extend(["--num-concurrency", str(concurrency)])

		# Add iteration limits (note: different parameter names than K8s BenchmarkJob)
		max_time = benchmark_config.get("max_time_per_iteration", 10)
		max_requests = benchmark_config.get("max_requests_per_iteration", 50)
		cmd.extend(["--max-time-per-run", str(max_time), "--max-requests-per-run", str(max_requests)])

		# Add additional request params (needs to be JSON string)
		additional_params = benchmark_config.get("additional_params", {})
		if additional_params:
			import json

			params_json = json.dumps(additional_params)
			cmd.extend(["--additional-request-params", params_json])

		print(f"[Benchmark] Command: {' '.join(cmd)}")

		# Run benchmark
		try:
			start_time = time.time()

			if self.verbose:
				# Stream output in real-time
				print(f"[Benchmark] Starting genai-bench (streaming output)...")
				process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

				stdout_lines = []
				for line in process.stdout:
					print(f"[genai-bench] {line.rstrip()}")
					stdout_lines.append(line)

				process.wait(timeout=timeout)
				result_returncode = process.returncode
				result_stdout = "".join(stdout_lines)
				result_stderr = ""
			else:
				# Capture output
				result = subprocess.run(
					cmd,
					capture_output=True,
					text=True,
					timeout=timeout,
					check=False,  # Don't raise exception on non-zero exit
				)
				result_returncode = result.returncode
				result_stdout = result.stdout
				result_stderr = result.stderr

			elapsed_time = time.time() - start_time

			print(f"[Benchmark] Completed in {elapsed_time:.1f}s")
			print(f"[Benchmark] Exit code: {result_returncode}")

			# Show genai-bench output for debugging (only if not in verbose mode)
			if not self.verbose:
				if result_stdout:
					print(f"[Benchmark] STDOUT:\n{result_stdout}")
				if result_stderr:
					print(f"[Benchmark] STDERR:\n{result_stderr}")

			if result_returncode != 0:
				return None

			# Parse results from output directory
			metrics = self._parse_results(output_dir)
			if metrics:
				metrics["elapsed_time"] = elapsed_time
				metrics["benchmark_name"] = benchmark_name
				return metrics
			else:
				print(f"[Benchmark] No results found in {output_dir}")
				return None

		except subprocess.TimeoutExpired:
			print(f"[Benchmark] Timeout after {timeout}s")
			return None
		except Exception as e:
			print(f"[Benchmark] Error running genai-bench: {e}")
			return None
		finally:
			# Only cleanup port forward if we set it up
			if need_cleanup:
				self.cleanup_port_forward()

	def _parse_results(self, output_dir: Path) -> Optional[Dict[str, Any]]:
		"""Parse benchmark results from output directory.

		Args:
		    output_dir: Directory containing benchmark results

		Returns:
		    Dict containing parsed metrics, or None if unavailable
		"""
		# Look for JSON results files in the experiment directory
		# genai-bench creates files like:
		# - experiment_metadata.json (metadata)
		# - D100_100_text-to-text_num_concurrency_1_time_9s.json (actual results per concurrency)
		result_files = list(output_dir.glob("D*.json"))  # Only get result files, not metadata
		if not result_files:
			print(f"[Benchmark] No result JSON files found in {output_dir}")
			print(f"[Benchmark] Directory contents:")
			if output_dir.exists():
				for item in output_dir.rglob("*"):
					print(f"  {item}")
			return None

		print(f"[Benchmark] Found {len(result_files)} result file(s)")

		try:
			# Parse all result files and aggregate metrics
			all_metrics = []
			for result_file in result_files:
				print(f"[Benchmark] Parsing {result_file.name}")
				with open(result_file, "r") as f:
					data = json.load(f)

				# genai-bench result structure: {"aggregated_metrics": {...}}
				if "aggregated_metrics" in data:
					all_metrics.append(data["aggregated_metrics"])
				else:
					print(f"[Benchmark] Warning: No aggregated_metrics in {result_file.name}")

			if not all_metrics:
				print(f"[Benchmark] No valid metrics found in result files")
				return None

			# Aggregate metrics across all concurrency levels
			# Use the best (lowest latency or highest throughput) from all runs
			aggregated = {
				"num_result_files": len(all_metrics),
				"concurrency_levels": [m.get("num_concurrency") for m in all_metrics],
				"raw_results": all_metrics,  # Keep all raw results for reference
			}

			# Extract key performance metrics
			# Average across all concurrency levels
			if all_metrics:
				# E2E Latency stats (mean across all concurrencies)
				e2e_latencies = [m["stats"]["e2e_latency"]["mean"] for m in all_metrics if "stats" in m]
				if e2e_latencies:
					aggregated["mean_e2e_latency"] = sum(e2e_latencies) / len(e2e_latencies)
					aggregated["min_e2e_latency"] = min(e2e_latencies)
					aggregated["max_e2e_latency"] = max(e2e_latencies)

				# P50, P90, P99 latencies (average across concurrencies)
				p50_latencies = [m["stats"]["e2e_latency"]["p50"] for m in all_metrics if "stats" in m]
				p90_latencies = [m["stats"]["e2e_latency"]["p90"] for m in all_metrics if "stats" in m]
				p99_latencies = [m["stats"]["e2e_latency"]["p99"] for m in all_metrics if "stats" in m]
				if p50_latencies:
					aggregated["p50_e2e_latency"] = sum(p50_latencies) / len(p50_latencies)
				if p90_latencies:
					aggregated["p90_e2e_latency"] = sum(p90_latencies) / len(p90_latencies)
				if p99_latencies:
					aggregated["p99_e2e_latency"] = sum(p99_latencies) / len(p99_latencies)

				# TTFT (Time to First Token) stats
				ttft_means = [m["stats"]["ttft"]["mean"] for m in all_metrics if "stats" in m]
				if ttft_means:
					aggregated["mean_ttft"] = sum(ttft_means) / len(ttft_means)

				# TPOT (Time Per Output Token) stats
				tpot_means = [m["stats"]["tpot"]["mean"] for m in all_metrics if "stats" in m]
				if tpot_means:
					aggregated["mean_tpot"] = sum(tpot_means) / len(tpot_means)

				# Throughput (tokens/s) - use max throughput achieved
				output_throughputs = [m.get("mean_output_throughput_tokens_per_s", 0) for m in all_metrics]
				if output_throughputs:
					aggregated["mean_output_throughput"] = sum(output_throughputs) / len(output_throughputs)
					aggregated["max_output_throughput"] = max(output_throughputs)

				total_throughputs = [m.get("mean_total_tokens_throughput_tokens_per_s", 0) for m in all_metrics]
				if total_throughputs:
					aggregated["mean_total_throughput"] = sum(total_throughputs) / len(total_throughputs)
					aggregated["max_total_throughput"] = max(total_throughputs)

				# Request stats
				total_requests = sum(m.get("num_requests", 0) for m in all_metrics)
				total_completed = sum(m.get("num_completed_requests", 0) for m in all_metrics)
				total_errors = sum(m.get("num_error_requests", 0) for m in all_metrics)
				aggregated["total_requests"] = total_requests
				aggregated["total_completed_requests"] = total_completed
				aggregated["total_error_requests"] = total_errors
				if total_requests > 0:
					aggregated["success_rate"] = total_completed / total_requests

			return aggregated

		except Exception as e:
			print(f"[Benchmark] Error parsing results: {e}")
			import traceback

			traceback.print_exc()
			return None

	def cleanup_results(self, task_name: str, experiment_id: int) -> bool:
		"""Clean up benchmark result files.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Experiment identifier

		Returns:
		    True if cleaned up successfully
		"""
		benchmark_name = f"{task_name}-exp{experiment_id}"
		output_dir = self.results_dir / benchmark_name

		if output_dir.exists():
			import shutil

			try:
				shutil.rmtree(output_dir)
				print(f"[Benchmark] Cleaned up results: {output_dir}")
				return True
			except Exception as e:
				print(f"[Benchmark] Error cleaning up results: {e}")
				return False
		return True
