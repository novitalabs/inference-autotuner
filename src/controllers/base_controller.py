"""
Base Controller Interface

Abstract base class for model deployment controllers.
Supports multiple deployment modes (OME/Kubernetes, Docker, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModelController(ABC):
	"""Abstract base class for model deployment controllers."""

	@abstractmethod
	def deploy_inference_service(
		self,
		task_name: str,
		experiment_id: int,
		namespace: str,
		model_name: str,
		runtime_name: str,
		parameters: Dict[str, Any],
	) -> Optional[str]:
		"""Deploy a model inference service with specified parameters.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    namespace: Namespace/resource group identifier
		    model_name: Model name/path
		    runtime_name: Runtime identifier (e.g., 'sglang')
		    parameters: Deployment parameters (tp_size, mem_frac, etc.)

		Returns:
		    Service identifier (name/ID) if successful, None otherwise
		"""
		pass

	@abstractmethod
	def wait_for_ready(self, service_id: str, namespace: str, timeout: int = 600, poll_interval: int = 10) -> bool:
		"""Wait for the inference service to become ready.

		Args:
		    service_id: Service identifier returned by deploy_inference_service
		    namespace: Namespace/resource group identifier
		    timeout: Maximum wait time in seconds
		    poll_interval: Polling interval in seconds

		Returns:
		    True if service is ready, False if timeout or error
		"""
		pass

	@abstractmethod
	def delete_inference_service(self, service_id: str, namespace: str) -> bool:
		"""Delete an inference service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace/resource group identifier

		Returns:
		    True if deleted successfully
		"""
		pass

	@abstractmethod
	def get_service_url(self, service_id: str, namespace: str) -> Optional[str]:
		"""Get the service URL/endpoint for the inference service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace/resource group identifier

		Returns:
		    Service URL/endpoint if available, None otherwise
		"""
		pass
