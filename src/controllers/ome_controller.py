"""
OME Deployment Controller

Manages the lifecycle of InferenceService resources for autotuning experiments.
"""

import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .base_controller import BaseModelController


class OMEController(BaseModelController):
    """Controller for managing OME InferenceService deployments."""

    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize the OME controller.

        Args:
            kubeconfig_path: Path to kubeconfig file. If None, uses in-cluster config.
        """
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_kube_config()
        except Exception:
            # Fallback to in-cluster config
            config.load_incluster_config()

        self.api_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.api_client)
        self.core_api = client.CoreV1Api(self.api_client)

        # Load templates
        template_dir = Path(__file__).parent.parent / "templates"
        with open(template_dir / "inference_service.yaml.j2") as f:
            self.isvc_template = Template(f.read())

    def create_namespace(self, namespace: str) -> bool:
        """Create namespace if it doesn't exist.

        Args:
            namespace: Namespace name

        Returns:
            True if created or already exists
        """
        try:
            self.core_api.read_namespace(namespace)
            print(f"Namespace '{namespace}' already exists")
            return True
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                ns_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.core_api.create_namespace(ns_body)
                print(f"Created namespace '{namespace}'")
                return True
            else:
                print(f"Error checking namespace: {e}")
                return False

    def deploy_inference_service(
        self,
        task_name: str,
        experiment_id: str,
        namespace: str,
        model_name: str,
        runtime_name: str,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Deploy an InferenceService with specified parameters.

        Args:
            task_name: Autotuning task name
            experiment_id: Unique experiment identifier
            namespace: K8s namespace
            model_name: Model name
            runtime_name: ServingRuntime name
            parameters: SGLang parameters (tp_size, mem_frac, etc.)

        Returns:
            InferenceService name if successful, None otherwise
        """
        isvc_name = f"{task_name}-exp{experiment_id}"

        # Render template
        rendered = self.isvc_template.render(
            namespace=namespace,
            isvc_name=isvc_name,
            task_name=task_name,
            experiment_id=experiment_id,
            model_name=model_name,
            runtime_name=runtime_name,
            **parameters
        )

        # Parse YAML (contains namespace + InferenceService)
        resources = list(yaml.safe_load_all(rendered))

        # Create namespace
        self.create_namespace(namespace)

        # Create InferenceService
        try:
            isvc_resource = resources[1]  # Second resource is the InferenceService
            self.custom_api.create_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=isvc_resource
            )
            print(f"Created InferenceService '{isvc_name}' in namespace '{namespace}'")
            return isvc_name
        except ApiException as e:
            print(f"Error creating InferenceService: {e}")
            return None

    def wait_for_ready(
        self,
        isvc_name: str,
        namespace: str,
        timeout: int = 600,
        poll_interval: int = 10
    ) -> bool:
        """Wait for InferenceService to become ready.

        Args:
            isvc_name: InferenceService name
            namespace: K8s namespace
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            True if ready, False if timeout or error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                isvc = self.custom_api.get_namespaced_custom_object(
                    group="ome.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=isvc_name
                )

                # Check status conditions
                status = isvc.get("status", {})
                conditions = status.get("conditions", [])

                for condition in conditions:
                    if condition.get("type") == "Ready":
                        if condition.get("status") == "True":
                            url = status.get("url", "N/A")
                            print(f"InferenceService '{isvc_name}' is ready! URL: {url}")
                            return True
                        elif condition.get("status") == "False":
                            reason = condition.get("reason", "Unknown")
                            message = condition.get("message", "No details")
                            print(f"InferenceService not ready - {reason}: {message}")

                elapsed = int(time.time() - start_time)
                print(f"Waiting for InferenceService '{isvc_name}' to be ready... ({elapsed}s)")
                time.sleep(poll_interval)

            except ApiException as e:
                print(f"Error checking InferenceService status: {e}")
                time.sleep(poll_interval)

        print(f"Timeout waiting for InferenceService '{isvc_name}' to be ready")
        return False

    def delete_inference_service(self, isvc_name: str, namespace: str) -> bool:
        """Delete an InferenceService.

        Args:
            isvc_name: InferenceService name
            namespace: K8s namespace

        Returns:
            True if deleted successfully
        """
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=isvc_name
            )
            print(f"Deleted InferenceService '{isvc_name}' from namespace '{namespace}'")
            return True
        except ApiException as e:
            if e.status == 404:
                print(f"InferenceService '{isvc_name}' not found (already deleted?)")
                return True
            else:
                print(f"Error deleting InferenceService: {e}")
                return False

    def get_service_url(self, isvc_name: str, namespace: str) -> Optional[str]:
        """Get the service URL for an InferenceService.

        Args:
            isvc_name: InferenceService name
            namespace: K8s namespace

        Returns:
            Service URL if available, None otherwise
        """
        try:
            isvc = self.custom_api.get_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=isvc_name
            )
            return isvc.get("status", {}).get("url")
        except ApiException as e:
            print(f"Error getting service URL: {e}")
            return None
