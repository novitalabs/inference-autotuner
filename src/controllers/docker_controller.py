"""
Docker Deployment Controller

Manages the lifecycle of model inference services using standalone Docker containers.
No Kubernetes required - direct Docker container management.
"""

import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional

try:
	import docker
	from docker.errors import DockerException, NotFound, APIError
except ImportError:
	docker = None

from .base_controller import BaseModelController


class DockerController(BaseModelController):
	"""Controller for managing standalone Docker container deployments."""

	def __init__(self, model_base_path: str = "/mnt/data/models", http_proxy: str = "", https_proxy: str = "", no_proxy: str = "", hf_token: str = ""):
		"""Initialize the Docker controller.

		Args:
		    model_base_path: Base path where models are stored on the host
		    http_proxy: HTTP proxy URL (optional)
		    https_proxy: HTTPS proxy URL (optional)
		    no_proxy: Comma-separated list of hosts to bypass proxy (optional)
		    hf_token: HuggingFace access token for gated models (optional)

		Note:
		    Container logs are retrieved before deletion and saved to task log file.
		    Containers are manually removed during cleanup phase.
		"""
		if docker is None:
			raise ImportError("Docker SDK for Python is not installed. " "Install it with: pip install docker")

		try:
			self.client = docker.from_env()
			# Test connection
			self.client.ping()
			print("[Docker] Successfully connected to Docker daemon")
		except DockerException as e:
			raise RuntimeError(f"Failed to connect to Docker daemon: {e}")

		self.model_base_path = Path(model_base_path)
		self.containers = {}  # Track containers by service_id

		# Store proxy settings
		self.http_proxy = http_proxy
		self.https_proxy = https_proxy
		self.no_proxy = no_proxy

		# Store HuggingFace token
		self.hf_token = hf_token

		if self.http_proxy or self.https_proxy:
			print(f"[Docker] Proxy configured - HTTP: {self.http_proxy or 'None'}, HTTPS: {self.https_proxy or 'None'}")
			if self.no_proxy:
				print(f"[Docker] No proxy for: {self.no_proxy}")

	def deploy_inference_service(
		self,
		task_name: str,
		experiment_id: int,
		namespace: str,
		model_name: str,
		runtime_name: str,
		parameters: Dict[str, Any],
		image_tag: Optional[str] = None,
	) -> Optional[str]:
		"""Deploy a model inference service using Docker.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    namespace: Namespace identifier (used for container naming)
		    model_name: Model name (HuggingFace model ID or local path)
		    runtime_name: Runtime identifier (e.g., 'sglang', 'vllm')
		    parameters: SGLang/runtime parameters (tp_size, mem_frac, etc.)
		    image_tag: Optional Docker image tag (e.g., 'v0.5.2-cu126')

		Returns:
		    Container ID if successful, None otherwise
		"""
		service_id = f"{namespace}-{task_name}-exp{experiment_id}"
		container_name = service_id

		# Determine if model_name is a local path or HuggingFace model ID
		# Local paths start with / or contain model_base_path
		use_local_model = False
		model_identifier = model_name
		volumes = {}

		if model_name.startswith("/") or "/" not in model_name:
			# Could be a local path - check if it exists
			if model_name.startswith("/"):
				model_path = Path(model_name)
			else:
				model_path = self.model_base_path / model_name

			if model_path.exists():
				# Local model exists - use volume mount
				use_local_model = True
				model_identifier = "/model"
				volumes = {str(model_path): {"bind": "/model", "mode": "ro"}}
				print(f"[Docker] Using local model at {model_path}")
			else:
				# Local path doesn't exist - fail early
				print(f"[Docker] ERROR: Local model path {model_path} does not exist")
				print(f"[Docker] Either:")
				print(f"[Docker]   1. Download the model to {model_path}")
				print(f"[Docker]   2. Use a HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
				return None
		else:
			# Contains / - likely a HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)
			print(f"[Docker] Using HuggingFace model ID: {model_name}")
			print(f"[Docker] Model will be downloaded from HuggingFace Hub if not cached")

		# Determine runtime image and command based on runtime_name
		runtime_config = self._get_runtime_config(runtime_name, parameters, image_tag)
		if not runtime_config:
			print(f"[Docker] Unsupported runtime: {runtime_name}")
			return None

		# Determine host port (avoid conflicts) - needed for both bridge and host networking
		host_port = self._find_available_port(8000, 8100)
		if not host_port:
			print(f"[Docker] Could not find available port in range 8000-8100")
			return None

		# Build command with model identifier and host port
		command_str = runtime_config["command"].format(model_path=model_identifier, port=host_port)

		# Add all parameters as command-line arguments
		for param_name, param_value in parameters.items():
			# Convert parameter name to CLI format (add -- prefix if not present)
			if not param_name.startswith("--"):
				cli_param = f"--{param_name}"
			else:
				cli_param = param_name
			command_str += f" {cli_param} {param_value}"

		command_list = command_str.split()

		# Determine GPU allocation
		# Look for tp-size or tp_size parameter for GPU count (default to 1)
		num_gpus = parameters.get("tp-size", parameters.get("tp_size", 1))
		if isinstance(num_gpus, (int, float)):
			num_gpus = int(num_gpus)
		else:
			num_gpus = 1

		# Build container configuration
		try:
			# Determine GPU devices
			gpu_devices = self._select_gpus(num_gpus)
			if not gpu_devices:
				print(f"[Docker] Failed to allocate {num_gpus} GPU(s)")
				return None

			print(f"[Docker] Deploying container '{container_name}'")
			print(f"[Docker] Image: {runtime_config['image']}")
			if use_local_model:
				print(f"[Docker] Model: {model_path} (local)")
			else:
				print(f"[Docker] Model: {model_name} (HuggingFace Hub)")
			print(f"[Docker] GPUs: {gpu_devices}")
			print(f"[Docker] Parameters: {parameters}")

			# Remove existing container if present
			try:
				old_container = self.client.containers.get(container_name)
				print(f"[Docker] Removing existing container '{container_name}'")
				old_container.remove(force=True)
			except NotFound:
				pass

			# Check if image exists locally, pull if not
			self._ensure_image_available(runtime_config["image"])

			# Prepare environment variables
			env_vars = {
				"MODEL_PATH": model_identifier,
				"HF_HOME": "/root/.cache/huggingface"  # Cache directory for downloaded models
				# Note: Don't set CUDA_VISIBLE_DEVICES as it conflicts with device_requests
			}

			# Add proxy settings if configured
			if self.http_proxy:
				env_vars["HTTP_PROXY"] = self.http_proxy
				env_vars["http_proxy"] = self.http_proxy
			if self.https_proxy:
				env_vars["HTTPS_PROXY"] = self.https_proxy
				env_vars["https_proxy"] = self.https_proxy
			if self.no_proxy:
				env_vars["NO_PROXY"] = self.no_proxy
				env_vars["no_proxy"] = self.no_proxy

			# Add HuggingFace token if configured
			if self.hf_token:
				env_vars["HF_TOKEN"] = self.hf_token
				env_vars["HUGGING_FACE_HUB_TOKEN"] = self.hf_token  # Alternative name some libraries use

			# Debug: Print env vars being passed to container
			print(f"[Docker] Environment variables to be set in container:")
			for key, value in env_vars.items():
				if "proxy" in key.lower() or "token" in key.lower():
					# Mask token value for security
					display_value = "***" if "token" in key.lower() and value else value
					print(f"[Docker]   {key}={display_value}")

			# Create and start container with host networking
			container = self.client.containers.run(
				image=runtime_config["image"],
				name=container_name,
				command=command_list,
				detach=True,
				device_requests=[docker.types.DeviceRequest(device_ids=gpu_devices, capabilities=[["gpu"]])],
				volumes=volumes,  # Use conditional volumes (empty for HuggingFace models)
				environment=env_vars,
				shm_size="16g",  # Shared memory for multi-process inference
				ipc_mode="host",  # Use host IPC namespace for shared memory
				network_mode="host",  # Use host network for better performance and compatibility
				remove=False,  # Don't auto-remove - we need to retrieve logs first
			)

			# Store container reference
			self.containers[service_id] = {
				"container": container,
				"host_port": host_port,
				"gpu_devices": gpu_devices,
			}

			print(f"[Docker] Container '{container_name}' started (ID: {container.short_id})")
			print(f"[Docker] Service URL: http://localhost:{host_port}")

			# Verify proxy settings in container - inspect via Docker API
			try:
				container.reload()  # Refresh container info
				container_env = container.attrs.get('Config', {}).get('Env', [])
				proxy_env = [env for env in container_env if 'proxy' in env.lower()]
				if proxy_env:
					print(f"[Docker] Proxy environment variables in container (from Docker API):")
					for env in proxy_env:
						print(f"[Docker]   {env}")
				else:
					print(f"[Docker] No proxy environment variables found in container (via Docker API)")
			except Exception as e:
				print(f"[Docker] Could not inspect container environment: {e}")

			return service_id

		except APIError as e:
			print(f"[Docker] Error creating container: {e}")
			return None
		except Exception as e:
			print(f"[Docker] Unexpected error: {e}")
			return None

	def wait_for_ready(self, service_id: str, namespace: str, timeout: int = 600, poll_interval: int = 5) -> bool:
		"""Wait for the Docker container service to become ready.

		Args:
		    service_id: Service identifier (container name)
		    namespace: Namespace identifier
		    timeout: Maximum wait time in seconds
		    poll_interval: Polling interval in seconds

		Returns:
		    True if service is ready, False if timeout or error
		"""
		if service_id not in self.containers:
			print(f"[Docker] Service '{service_id}' not found")
			return False

		container_info = self.containers[service_id]
		container = container_info["container"]
		host_port = container_info["host_port"]
		health_url = f"http://localhost:{host_port}/health"

		start_time = time.time()
		print(f"[Docker] Waiting for service to be ready at {health_url}...")

		# Track consecutive failures for crash-loop detection
		consecutive_exits = 0
		max_consecutive_exits = 3

		# Track log snapshots - capture logs at key intervals for debugging
		log_snapshot_intervals = [60, 120, 300]  # Capture logs at 1min, 2min, 5min
		next_snapshot_idx = 0

		while time.time() - start_time < timeout:
			try:
				# Check container status
				container.reload()

				# Handle different container states
				if container.status == "running":
					# Container is running, try health check
					consecutive_exits = 0  # Reset counter

					try:
						response = requests.get(health_url, timeout=5)
						if response.status_code == 200:
							print(f"[Docker] Service is ready! URL: http://localhost:{host_port}")
							return True
					except requests.RequestException:
						# Health endpoint not ready yet, continue waiting
						pass

				elif container.status in ["exited", "dead"]:
					# Container has stopped - this is a failure
					consecutive_exits += 1
					print(f"[Docker] Container status: {container.status} (attempt {consecutive_exits}/{max_consecutive_exits})")

					# Get exit code for more information
					exit_code = container.attrs.get('State', {}).get('ExitCode', 'unknown')
					print(f"[Docker] Container exit code: {exit_code}")

					# Print logs to help diagnose the issue
					try:
						# Retrieve ALL logs to diagnose startup issues (not just last 100 lines)
						logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
						print(f"[Docker] Container logs:\n{logs}")
					except Exception as e:
						print(f"[Docker] Could not retrieve container logs: {e}")

					# If container exits multiple times quickly, it's crash-looping - fail immediately
					if consecutive_exits >= max_consecutive_exits:
						print(f"[Docker] Container is crash-looping, giving up")
						return False

					# Container exited, likely due to error - fail immediately
					print(f"[Docker] Container stopped unexpectedly, deployment failed")
					return False

				elif container.status in ["removing", "paused"]:
					# Container is being removed or paused - this is a failure
					print(f"[Docker] Container status: {container.status} - deployment failed")
					return False

				elif container.status in ["created", "restarting"]:
					# Container is starting or restarting - keep waiting
					print(f"[Docker] Container status: {container.status} - waiting for running state...")

				else:
					# Unknown status
					print(f"[Docker] Container status: {container.status} (unknown state)")

			except NotFound:
				# Container has been auto-removed (because it exited with remove=True)
				print(f"[Docker] Container was automatically removed after exiting")
				print(f"[Docker] This typically means the container failed to start")
				print(f"[Docker] Check that the model path exists and the runtime parameters are correct")
				return False
			except Exception as e:
				print(f"[Docker] Error checking service status: {e}")
				# If we can't check status, the container might be gone
				return False

			elapsed = int(time.time() - start_time)

			# Capture log snapshots at key intervals for debugging long startups
			if next_snapshot_idx < len(log_snapshot_intervals):
				if elapsed >= log_snapshot_intervals[next_snapshot_idx]:
					print(f"\n[Docker] === Log Snapshot at {elapsed}s ===")
					try:
						snapshot_logs = container.logs(tail=50, stdout=True, stderr=True).decode("utf-8", errors="replace")
						print(snapshot_logs)
						print(f"[Docker] === End Snapshot ===\n")
					except Exception as e:
						print(f"[Docker] Could not capture log snapshot: {e}")
					next_snapshot_idx += 1

			print(f"[Docker] Waiting for service... ({elapsed}s)")
			time.sleep(poll_interval)

		# Timeout reached
		print(f"[Docker] Timeout waiting for service '{service_id}' to be ready after {timeout}s")

		# Print final container logs for debugging
		try:
			container.reload()
			print(f"[Docker] Final container status: {container.status}")
			# Retrieve ALL logs to diagnose startup issues (not just last 100 lines)
			logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
			print(f"[Docker] Container logs:\n{logs}")
		except Exception as e:
			print(f"[Docker] Could not retrieve final container state: {e}")

		return False

	def delete_inference_service(self, service_id: str, namespace: str) -> bool:
		"""Delete a Docker container service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier

		Returns:
		    True if deleted successfully
		"""
		if service_id not in self.containers:
			print(f"[Docker] Service '{service_id}' not found (already deleted?)")
			return True

		try:
			container_info = self.containers[service_id]
			container = container_info["container"]

			print(f"[Docker] Stopping and removing container '{service_id}'...")

			# Stop the container first
			try:
				container.stop(timeout=10)
			except Exception as e:
				print(f"[Docker] Error stopping container (may already be stopped): {e}")

			# Now remove the container
			try:
				container.remove(force=True)
			except Exception as e:
				print(f"[Docker] Error removing container: {e}")

			# Release GPU tracking (if implemented)
			del self.containers[service_id]

			print(f"[Docker] Container '{service_id}' removed")
			return True

		except NotFound:
			print(f"[Docker] Container '{service_id}' not found (already deleted?)")
			del self.containers[service_id]
			return True
		except Exception as e:
			print(f"[Docker] Error deleting container: {e}")
			return False

	def get_service_url(self, service_id: str, namespace: str) -> Optional[str]:
		"""Get the service URL for a Docker container.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier

		Returns:
		    Service URL if available, None otherwise
		"""
		if service_id not in self.containers:
			return None

		host_port = self.containers[service_id]["host_port"]
		return f"http://localhost:{host_port}"

	def get_container_logs(self, service_id: str, namespace: str, tail: int = 1000) -> Optional[str]:
		"""Get logs from a Docker container.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier
		    tail: Number of lines to retrieve (default: 1000, 0 for all)

		Returns:
		    Container logs as string, None if container not found
		"""
		if service_id not in self.containers:
			print(f"[Docker] Service '{service_id}' not found, cannot retrieve logs")
			return None

		try:
			container = self.containers[service_id]["container"]

			# Get logs (both stdout and stderr)
			if tail > 0:
				logs = container.logs(tail=tail, stdout=True, stderr=True).decode("utf-8", errors="replace")
			else:
				logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

			return logs
		except Exception as e:
			print(f"[Docker] Error retrieving logs for '{service_id}': {e}")
			return None

	def _get_runtime_config(
		self, runtime_name: str, parameters: Dict[str, Any], image_tag: Optional[str] = None
	) -> Optional[Dict[str, str]]:
		"""Get Docker image and command configuration for a runtime.

		Args:
		    runtime_name: Runtime identifier
		    parameters: Runtime parameters (unused, kept for compatibility)
		    image_tag: Optional Docker image tag to override default

		Returns:
		    Dictionary with 'image' and 'command' keys, or None if unsupported
		"""
		# Map runtime names to Docker images and base commands
		# Base command only includes essential fixed parameters
		# Dynamic parameters are added by the caller
		runtime_configs = {
			"sglang": {
				"image": "lmsysorg/sglang:v0.5.2-cu126",
				"command": "python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --port {port}",
			},
			"vllm": {
				"image": "vllm/vllm-openai:latest",
				"command": "python3 -m vllm.entrypoints.openai.api_server --model {model_path} --host 0.0.0.0 --port {port}",
			},
		}

		# Try exact match or prefix match
		config = None
		for key, cfg in runtime_configs.items():
			if runtime_name.lower().startswith(key):
				config = cfg.copy()
				break

		if not config:
			return None

		# Override image tag if provided
		if image_tag:
			# Extract base image name (before colon)
			base_image = config["image"].split(":")[0]
			config["image"] = f"{base_image}:{image_tag}"
			print(f"[Docker] Using custom image tag: {config['image']}")

		return config

	def _ensure_image_available(self, image_name: str) -> bool:
		"""Ensure Docker image is available locally, pull if not.

		Args:
		    image_name: Full image name with tag (e.g., 'lmsysorg/sglang:v0.5.2-cu126')

		Returns:
		    True if image is available, False if pull failed
		"""
		try:
			# Check if image exists locally
			self.client.images.get(image_name)
			print(f"[Docker] Image '{image_name}' found in local cache")
			return True
		except docker.errors.ImageNotFound:
			# Image not found locally, need to pull
			print(f"[Docker] Image '{image_name}' not found locally")
			print(f"[Docker] Pulling image (this may take several minutes)...")

			try:
				# Split image name and tag for pull API
				if ":" in image_name:
					repository, tag = image_name.rsplit(":", 1)
				else:
					repository = image_name
					tag = "latest"

				# Pull with progress tracking
				last_status = {}
				for line in self.client.api.pull(repository, tag=tag, stream=True, decode=True):
					# Each line is a dict with status, progressDetail, etc.
					if "status" in line:
						status = line["status"]
						layer_id = line.get("id", "")

						# Show progress for downloading/extracting layers
						if "progressDetail" in line and line["progressDetail"]:
							progress = line["progressDetail"]
							current = progress.get("current", 0)
							total = progress.get("total", 0)

							if total > 0:
								percent = (current / total) * 100
								# Update status for this layer
								last_status[layer_id] = f"{status}: {percent:.1f}%"
							else:
								last_status[layer_id] = status
						else:
							# Status without progress (e.g., "Pull complete")
							if layer_id:
								last_status[layer_id] = status

						# Print summary of active layers periodically
						if layer_id and status in ["Downloading", "Extracting"]:
							# Count layers in each state
							downloading = sum(1 for s in last_status.values() if "Downloading" in s)
							extracting = sum(1 for s in last_status.values() if "Extracting" in s)
							complete = sum(1 for s in last_status.values() if "complete" in s)

							# Show compact progress summary
							print(
								f"\r[Docker] Progress: {complete} complete, {downloading} downloading, {extracting} extracting",
								end="",
								flush=True,
							)

				# Final newline after progress
				print()
				print(f"[Docker] Successfully pulled '{image_name}'")
				return True

			except Exception as e:
				print(f"\n[Docker] Error pulling image: {e}")
				return False
		except Exception as e:
			print(f"[Docker] Error checking image: {e}")
			return False

	def _select_gpus(self, num_gpus: int) -> Optional[list]:
		"""Select GPU devices for the container.

		Args:
		    num_gpus: Number of GPUs required

		Returns:
		    List of GPU device IDs (as strings), or None if not enough GPUs
		"""
		# TODO: Implement proper GPU tracking and allocation
		# For now, use simple sequential allocation
		try:
			# Query available GPUs using nvidia-smi
			import subprocess

			result = subprocess.run(
				["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
				capture_output=True,
				text=True,
				check=True,
			)

			gpus = []
			for line in result.stdout.strip().split("\n"):
				if line:
					gpu_id, free_mem = line.split(",")
					gpus.append((gpu_id.strip(), int(free_mem.strip())))

			# Sort by free memory (descending)
			gpus.sort(key=lambda x: x[1], reverse=True)

			if len(gpus) < num_gpus:
				print(f"[Docker] Not enough GPUs available: {len(gpus)} < {num_gpus}")
				return None

			# Select top N GPUs with most free memory
			selected = [gpu[0] for gpu in gpus[:num_gpus]]
			return selected

		except FileNotFoundError:
			print("[Docker] nvidia-smi not found. Using default GPU allocation.")
			# Fallback: use first N GPUs
			return [str(i) for i in range(num_gpus)]
		except Exception as e:
			print(f"[Docker] Error selecting GPUs: {e}")
			# Fallback
			return [str(i) for i in range(num_gpus)]

	def _find_available_port(self, start_port: int, end_port: int) -> Optional[int]:
		"""Find an available port in the specified range.

		Args:
		    start_port: Start of port range
		    end_port: End of port range

		Returns:
		    Available port number, or None if no ports available
		"""
		import socket

		for port in range(start_port, end_port + 1):
			try:
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					s.bind(("", port))
					return port
			except OSError:
				continue

		return None
