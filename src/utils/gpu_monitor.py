"""
GPU Monitoring and Management Utilities

Provides comprehensive GPU tracking, allocation, and monitoring capabilities
for the LLM Inference Autotuner.
"""

import subprocess
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Comprehensive GPU information."""
    index: int
    name: str
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    memory_usage_percent: float
    utilization_percent: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    compute_mode: str
    processes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def is_available(self) -> bool:
        """Check if GPU is available for new workload."""
        # Consider available if utilization < 50% and memory usage < 80%
        return (self.utilization_percent < 50 and
                self.memory_usage_percent < 80)

    @property
    def available_memory_mb(self) -> int:
        """Get available memory in MB."""
        return self.memory_free_mb

    @property
    def score(self) -> float:
        """
        Calculate availability score for GPU allocation.
        Higher score = better for allocation.
        Range: 0.0 (fully used) to 1.0 (completely free)
        """
        memory_score = self.memory_free_mb / self.memory_total_mb
        util_score = (100 - self.utilization_percent) / 100
        # Weighted average: memory (60%), utilization (40%)
        return 0.6 * memory_score + 0.4 * util_score


@dataclass
class GPUSnapshot:
    """Snapshot of all GPUs at a point in time."""
    timestamp: datetime
    gpus: List[GPUInfo]
    total_gpus: int
    available_gpus: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "total_gpus": self.total_gpus,
            "available_gpus": self.available_gpus
        }


class GPUMonitor:
    """Monitor and manage GPU resources."""

    def __init__(self):
        self._cache: Optional[GPUSnapshot] = None
        self._cache_ttl = 2.0  # Cache for 2 seconds
        self._last_update = 0.0

    def is_available(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_gpu_count(self) -> int:
        """Get total number of GPUs."""
        if not self.is_available():
            return 0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return len([line for line in result.stdout.strip().split("\n") if line])
            return 0
        except Exception as e:
            logger.error(f"Failed to get GPU count: {e}")
            return 0

    def query_gpus(self, use_cache: bool = True) -> Optional[GPUSnapshot]:
        """
        Query all GPU information.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            GPUSnapshot or None if query fails
        """
        # Return cache if valid and requested
        now = time.time()
        if use_cache and self._cache and (now - self._last_update) < self._cache_ttl:
            return self._cache

        if not self.is_available():
            logger.warning("nvidia-smi not available")
            return None

        try:
            # Query comprehensive GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
                    "utilization.gpu,temperature.gpu,power.draw,power.limit,compute_mode",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return None

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 11:
                    index = int(parts[0])

                    # Query processes for this GPU
                    processes = self._query_gpu_processes(index)

                    memory_total = int(parts[3])
                    memory_used = int(parts[4])

                    gpu_info = GPUInfo(
                        index=index,
                        name=parts[1],
                        uuid=parts[2],
                        memory_total_mb=memory_total,
                        memory_used_mb=memory_used,
                        memory_free_mb=int(parts[5]),
                        memory_usage_percent=round(memory_used / memory_total * 100, 1) if memory_total > 0 else 0,
                        utilization_percent=int(parts[6]) if parts[6] != "N/A" else 0,
                        temperature_c=int(parts[7]) if parts[7] != "N/A" else 0,
                        power_draw_w=float(parts[8]) if parts[8] != "N/A" else 0.0,
                        power_limit_w=float(parts[9]) if parts[9] != "N/A" else 0.0,
                        compute_mode=parts[10],
                        processes=processes
                    )
                    gpus.append(gpu_info)

            snapshot = GPUSnapshot(
                timestamp=datetime.now(),
                gpus=gpus,
                total_gpus=len(gpus),
                available_gpus=sum(1 for gpu in gpus if gpu.is_available)
            )

            # Update cache
            self._cache = snapshot
            self._last_update = now

            return snapshot

        except Exception as e:
            logger.error(f"Failed to query GPUs: {e}")
            return None

    def _query_gpu_processes(self, gpu_index: int) -> List[Dict[str, Any]]:
        """Query processes running on specific GPU."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_index}"
                ],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode != 0:
                return []

            processes = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    processes.append({
                        "pid": int(parts[0]),
                        "used_memory_mb": int(parts[1])
                    })

            return processes

        except Exception as e:
            logger.debug(f"Failed to query GPU {gpu_index} processes: {e}")
            return []

    def get_available_gpus(self,
                          min_memory_mb: Optional[int] = None,
                          max_utilization: int = 50) -> List[int]:
        """
        Get list of available GPU indices.

        Args:
            min_memory_mb: Minimum free memory required (MB)
            max_utilization: Maximum GPU utilization allowed (%)

        Returns:
            List of GPU indices sorted by availability score (best first)
        """
        snapshot = self.query_gpus()
        if not snapshot:
            return []

        available = []
        for gpu in snapshot.gpus:
            # Check utilization
            if gpu.utilization_percent > max_utilization:
                continue

            # Check memory if specified
            if min_memory_mb and gpu.available_memory_mb < min_memory_mb:
                continue

            available.append(gpu)

        # Sort by availability score (best first)
        available.sort(key=lambda g: g.score, reverse=True)

        return [gpu.index for gpu in available]

    def allocate_gpus(self,
                     count: int,
                     min_memory_mb: Optional[int] = None) -> Tuple[List[int], bool]:
        """
        Allocate specified number of GPUs.

        Args:
            count: Number of GPUs to allocate
            min_memory_mb: Minimum memory required per GPU (MB)

        Returns:
            Tuple of (allocated_gpu_indices, success)
        """
        available = self.get_available_gpus(min_memory_mb=min_memory_mb)

        if len(available) < count:
            logger.warning(
                f"Requested {count} GPUs but only {len(available)} available "
                f"(min_memory_mb={min_memory_mb})"
            )
            return ([], False)

        # Allocate best GPUs
        allocated = available[:count]
        logger.info(f"Allocated GPUs: {allocated}")

        return (allocated, True)

    def get_gpu_info(self, gpu_index: int) -> Optional[GPUInfo]:
        """Get information for specific GPU."""
        snapshot = self.query_gpus()
        if not snapshot:
            return None

        for gpu in snapshot.gpus:
            if gpu.index == gpu_index:
                return gpu

        return None

    def monitor_gpus(self,
                    gpu_indices: List[int],
                    duration_seconds: float,
                    interval_seconds: float = 1.0) -> List[GPUSnapshot]:
        """
        Monitor specific GPUs over time.

        Args:
            gpu_indices: List of GPU indices to monitor
            duration_seconds: How long to monitor
            interval_seconds: Sampling interval

        Returns:
            List of GPU snapshots
        """
        snapshots = []
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            snapshot = self.query_gpus(use_cache=False)
            if snapshot:
                # Filter to requested GPUs
                filtered_gpus = [gpu for gpu in snapshot.gpus if gpu.index in gpu_indices]
                filtered_snapshot = GPUSnapshot(
                    timestamp=snapshot.timestamp,
                    gpus=filtered_gpus,
                    total_gpus=len(filtered_gpus),
                    available_gpus=sum(1 for gpu in filtered_gpus if gpu.is_available)
                )
                snapshots.append(filtered_snapshot)

            time.sleep(interval_seconds)

        return snapshots

    def get_summary_stats(self, snapshots: List[GPUSnapshot]) -> Dict[str, Any]:
        """
        Calculate summary statistics from monitoring snapshots.

        Args:
            snapshots: List of GPU snapshots

        Returns:
            Dictionary with summary statistics
        """
        if not snapshots:
            return {}

        # Aggregate per-GPU statistics
        gpu_stats = {}

        for snapshot in snapshots:
            for gpu in snapshot.gpus:
                if gpu.index not in gpu_stats:
                    gpu_stats[gpu.index] = {
                        "name": gpu.name,
                        "utilization": [],
                        "memory_used": [],
                        "memory_usage_percent": [],
                        "temperature": [],
                        "power_draw": []
                    }

                gpu_stats[gpu.index]["utilization"].append(gpu.utilization_percent)
                gpu_stats[gpu.index]["memory_used"].append(gpu.memory_used_mb)
                gpu_stats[gpu.index]["memory_usage_percent"].append(gpu.memory_usage_percent)
                gpu_stats[gpu.index]["temperature"].append(gpu.temperature_c)
                gpu_stats[gpu.index]["power_draw"].append(gpu.power_draw_w)

        # Calculate statistics
        summary = {}
        for gpu_index, stats in gpu_stats.items():
            summary[gpu_index] = {
                "name": stats["name"],
                "utilization": {
                    "min": min(stats["utilization"]),
                    "max": max(stats["utilization"]),
                    "mean": sum(stats["utilization"]) / len(stats["utilization"]),
                    "samples": len(stats["utilization"])
                },
                "memory_used_mb": {
                    "min": min(stats["memory_used"]),
                    "max": max(stats["memory_used"]),
                    "mean": sum(stats["memory_used"]) / len(stats["memory_used"])
                },
                "memory_usage_percent": {
                    "min": min(stats["memory_usage_percent"]),
                    "max": max(stats["memory_usage_percent"]),
                    "mean": sum(stats["memory_usage_percent"]) / len(stats["memory_usage_percent"])
                },
                "temperature_c": {
                    "min": min(stats["temperature"]),
                    "max": max(stats["temperature"]),
                    "mean": sum(stats["temperature"]) / len(stats["temperature"])
                },
                "power_draw_w": {
                    "min": min(stats["power_draw"]),
                    "max": max(stats["power_draw"]),
                    "mean": sum(stats["power_draw"]) / len(stats["power_draw"])
                }
            }

        return {
            "monitoring_duration_seconds": (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds(),
            "sample_count": len(snapshots),
            "gpu_stats": summary
        }


# Global GPU monitor instance
_gpu_monitor = None


def get_gpu_monitor() -> GPUMonitor:
    """Get global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor
