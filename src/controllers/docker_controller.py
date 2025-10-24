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

	def __init__(self, model_base_path: str = "/mnt/data/models"):
		"""Initialize the Docker controller.

		Args:
		    model_base_path: Base path where models are stored on the host

		Note:
		    Containers are automatically removed after they stop for automatic cleanup.
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
		    model_name: Model name (used to find model path)
		    runtime_name: Runtime identifier (e.g., 'sglang', 'vllm')
		    parameters: SGLang/runtime parameters (tp_size, mem_frac, etc.)
		    image_tag: Optional Docker image tag (e.g., 'v0.5.2-cu126')

		Returns:
		    Container ID if successful, None otherwise
		"""
		service_id = f"{namespace}-{task_name}-exp{experiment_id}"
		container_name = service_id

		# Determine model path
		model_path = self.model_base_path / model_name
		if not model_path.exists():
			print(f"[Docker] Warning: Model path {model_path} does not exist on host")
			print(f"[Docker] Container will attempt to use model path: {model_path}")

		# Determine runtime image and command based on runtime_name
		runtime_config = self._get_runtime_config(runtime_name, parameters, image_tag)
		if not runtime_config:
			print(f"[Docker] Unsupported runtime: {runtime_name}")
			return None

		# Extract parameters
		# Build command with arbitrary parameters
		command_str = runtime_config["command"].format(model_path=f"/model")

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
			print(f"[Docker] Model: {model_path}")
			print(f"[Docker] GPUs: {gpu_devices}")
			print(f"[Docker] Parameters: {parameters}")

			# Remove existing container if present
			try:
				old_container = self.client.containers.get(container_name)
				print(f"[Docker] Removing existing container '{container_name}'")
				old_container.remove(force=True)
			except NotFound:
				pass

			# Determine host port (avoid conflicts)
			host_port = self._find_available_port(8000, 8100)
			if not host_port:
				print("[Docker] No available ports in range 8000-8100")
				return None

			# Check if image exists locally, pull if not
			self._ensure_image_available(runtime_config["image"])

			# Create and start container
			container = self.client.containers.run(
				image=runtime_config["image"],
				name=container_name,
				command=command_list,
				detach=True,
				device_requests=[docker.types.DeviceRequest(device_ids=gpu_devices, capabilities=[["gpu"]])],
				ports={"8080/tcp": host_port},
				volumes={str(model_path): {"bind": "/model", "mode": "ro"}},
				environment={
					"MODEL_PATH": "/model"
					# Note: Don't set CUDA_VISIBLE_DEVICES as it conflicts with device_requests
				},
				shm_size="16g",  # Shared memory for multi-process inference
				remove=True,  # Auto-remove container after stop for automatic cleanup
			)

			# Store container reference
			self.containers[service_id] = {
				"container": container,
				"host_port": host_port,
				"gpu_devices": gpu_devices,
			}

			print(f"[Docker] Container '{container_name}' started (ID: {container.short_id})")
			print(f"[Docker] Service URL: http://localhost:{host_port}")

			return service_id

		except APIError as e:
			print(f"[Docker] Error creating container: {e}")
			return None
		except Exception as e:
			print(f"[Docker] Unexpected error: {e}")
			return None

	def wait_for_ready(self, service_id: str, namespace: str, timeout: int = 600, poll_interval: int = 10) -> bool:
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

		while time.time() - start_time < timeout:
			try:
				# Check container status
				container.reload()
				if container.status != "running":
					print(f"[Docker] Container status: {container.status}")
					if container.status == "exited":
						logs = container.logs(tail=50).decode("utf-8")
						print(f"[Docker] Container logs:\n{logs}")
						return False
					time.sleep(poll_interval)
					continue

				# Try health check
				response = requests.get(health_url, timeout=5)
				if response.status_code == 200:
					print(f"[Docker] Service is ready! URL: http://localhost:{host_port}")
					return True

			except requests.RequestException:
				# Service not ready yet
				pass
			except Exception as e:
				print(f"[Docker] Error checking service status: {e}")

			elapsed = int(time.time() - start_time)
			print(f"[Docker] Waiting for service... ({elapsed}s)")
			time.sleep(poll_interval)

		print(f"[Docker] Timeout waiting for service '{service_id}' to be ready")

		# Print container logs for debugging
		try:
			logs = container.logs(tail=100).decode("utf-8")
			print(f"[Docker] Container logs:\n{logs}")
		except Exception as e:
			print(f"[Docker] Could not retrieve container logs: {e}")

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
			container.stop(timeout=10)
			container.remove()

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
				"command": "python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --port 8080",
			},
			"vllm": {
				"image": "vllm/vllm-openai:latest",
				"command": "python3 -m vllm.entrypoints.openai.api_server --model {model_path} --host 0.0.0.0 --port 8080",
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
