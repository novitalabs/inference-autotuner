"""
GenAI-Bench Wrapper

Manages BenchmarkJob resources and collects metrics.
"""

import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Template
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class BenchmarkController:
    """Controller for managing OME BenchmarkJob resources."""

    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize the benchmark controller.

        Args:
            kubeconfig_path: Path to kubeconfig file. If None, uses in-cluster config.
        """
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_kube_config()
        except Exception:
            config.load_incluster_config()

        self.api_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.api_client)
        self.batch_api = client.BatchV1Api(self.api_client)

        # Load templates
        template_dir = Path(__file__).parent.parent / "templates"
        with open(template_dir / "benchmark_job.yaml.j2") as f:
            self.benchmark_template = Template(f.read())

    def create_benchmark_job(
        self,
        task_name: str,
        experiment_id: str,
        namespace: str,
        isvc_name: str,
        benchmark_config: Dict[str, Any]
    ) -> Optional[str]:
        """Create a BenchmarkJob to evaluate an InferenceService.

        Args:
            task_name: Autotuning task name
            experiment_id: Unique experiment identifier
            namespace: K8s namespace
            isvc_name: InferenceService name to benchmark
            benchmark_config: Benchmark configuration from input JSON

        Returns:
            BenchmarkJob name if successful, None otherwise
        """
        benchmark_name = f"{task_name}-bench{experiment_id}"

        # Render template
        rendered = self.benchmark_template.render(
            benchmark_name=benchmark_name,
            namespace=namespace,
            task_name=task_name,
            experiment_id=experiment_id,
            isvc_name=isvc_name,
            task_type=benchmark_config.get("task", "text-to-text"),
            traffic_scenarios=benchmark_config.get("traffic_scenarios", ["D(100,100)"]),
            num_concurrency=benchmark_config.get("num_concurrency", [1]),
            max_time_per_iteration=benchmark_config.get("max_time_per_iteration", 15),
            max_requests_per_iteration=benchmark_config.get("max_requests_per_iteration", 100),
            additional_params=benchmark_config.get("additional_params", {})
        )

        # Parse YAML
        benchmark_resource = yaml.safe_load(rendered)

        # Create BenchmarkJob
        try:
            self.custom_api.create_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="benchmarkjobs",
                body=benchmark_resource
            )
            print(f"Created BenchmarkJob '{benchmark_name}' in namespace '{namespace}'")
            return benchmark_name
        except ApiException as e:
            print(f"Error creating BenchmarkJob: {e}")
            return None

    def wait_for_completion(
        self,
        benchmark_name: str,
        namespace: str,
        timeout: int = 1800,
        poll_interval: int = 15
    ) -> bool:
        """Wait for BenchmarkJob to complete.

        Args:
            benchmark_name: BenchmarkJob name
            namespace: K8s namespace
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            True if completed successfully, False if timeout or failed
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                job = self.custom_api.get_namespaced_custom_object(
                    group="ome.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="benchmarkjobs",
                    name=benchmark_name
                )

                # Check status conditions
                status = job.get("status", {})
                conditions = status.get("conditions", [])

                for condition in conditions:
                    if condition.get("type") == "Complete":
                        if condition.get("status") == "True":
                            print(f"BenchmarkJob '{benchmark_name}' completed successfully")
                            return True
                    elif condition.get("type") == "Failed":
                        if condition.get("status") == "True":
                            reason = condition.get("reason", "Unknown")
                            message = condition.get("message", "No details")
                            print(f"BenchmarkJob failed - {reason}: {message}")
                            return False

                elapsed = int(time.time() - start_time)
                print(f"Waiting for BenchmarkJob '{benchmark_name}' to complete... ({elapsed}s)")
                time.sleep(poll_interval)

            except ApiException as e:
                print(f"Error checking BenchmarkJob status: {e}")
                time.sleep(poll_interval)

        print(f"Timeout waiting for BenchmarkJob '{benchmark_name}' to complete")
        return False

    def get_benchmark_results(
        self,
        benchmark_name: str,
        namespace: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve benchmark results from BenchmarkJob status.

        Args:
            benchmark_name: BenchmarkJob name
            namespace: K8s namespace

        Returns:
            Dict containing benchmark metrics, or None if unavailable
        """
        try:
            job = self.custom_api.get_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="benchmarkjobs",
                name=benchmark_name
            )

            status = job.get("status", {})
            results = status.get("results", {})

            if not results:
                print(f"No results found for BenchmarkJob '{benchmark_name}'")
                return None

            # Extract key metrics
            metrics = {
                "benchmark_name": benchmark_name,
                "status": status,
                "results": results
            }

            print(f"Retrieved results for BenchmarkJob '{benchmark_name}'")
            return metrics

        except ApiException as e:
            print(f"Error getting benchmark results: {e}")
            return None

    def delete_benchmark_job(self, benchmark_name: str, namespace: str) -> bool:
        """Delete a BenchmarkJob.

        Args:
            benchmark_name: BenchmarkJob name
            namespace: K8s namespace

        Returns:
            True if deleted successfully
        """
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="ome.io",
                version="v1beta1",
                namespace=namespace,
                plural="benchmarkjobs",
                name=benchmark_name
            )
            print(f"Deleted BenchmarkJob '{benchmark_name}' from namespace '{namespace}'")
            return True
        except ApiException as e:
            if e.status == 404:
                print(f"BenchmarkJob '{benchmark_name}' not found (already deleted?)")
                return True
            else:
                print(f"Error deleting BenchmarkJob: {e}")
                return False
