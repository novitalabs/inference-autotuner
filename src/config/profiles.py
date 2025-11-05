"""
Built-in configuration profiles for common use cases.
"""

from .layers import ConfigLayer
from .factory import TaskConfigFactory


# High throughput preset
HIGH_THROUGHPUT_LAYERS = [
    ConfigLayer(
        name="high-throughput-params",
        data={
            "optimization": {
                "objective": "maximize_throughput",
                "max_iterations": 20
            },
            "parameters": {
                "mem-fraction-static": [0.95],
                "tp-size": [4, 8],
            },
            "benchmark": {
                "num_concurrency": [16, 32, 64]
            }
        }
    )
]

# Low latency preset
LOW_LATENCY_LAYERS = [
    ConfigLayer(
        name="low-latency-params",
        data={
            "optimization": {
                "objective": "minimize_latency",
                "max_iterations": 15
            },
            "parameters": {
                "mem-fraction-static": [0.7, 0.8],
                "tp-size": [1, 2],
            },
            "benchmark": {
                "num_concurrency": [1, 4, 8]
            }
        }
    )
]

# Quick test preset (minimal configuration for fast validation)
QUICK_TEST_LAYERS = [
    ConfigLayer(
        name="quick-test-params",
        data={
            "optimization": {
                "max_iterations": 2
            },
            "parameters": {
                "tp-size": [1],
                "mem-fraction-static": [0.85]
            },
            "benchmark": {
                "num_concurrency": [1],
                "traffic_scenarios": ["D(100,100)"]
            }
        }
    )
]

# Production preset (comprehensive testing with SLO constraints)
PRODUCTION_LAYERS = [
    ConfigLayer(
        name="production-params",
        data={
            "optimization": {
                "max_iterations": 30,
                "timeout_per_iteration": 900
            },
            "slo": {
                "ttft": {"threshold": 1.0, "weight": 2.0},
                "tpot": {"threshold": 0.05, "weight": 2.0},
                "latency": {
                    "p90": {
                        "threshold": 5.0,
                        "weight": 3.0,
                        "hard_fail": True,
                        "fail_ratio": 0.2
                    }
                }
            }
        }
    )
]


def register_builtin_profiles():
    """Register all built-in configuration profiles."""
    TaskConfigFactory.register_profile("high-throughput", HIGH_THROUGHPUT_LAYERS)
    TaskConfigFactory.register_profile("low-latency", LOW_LATENCY_LAYERS)
    TaskConfigFactory.register_profile("quick-test", QUICK_TEST_LAYERS)
    TaskConfigFactory.register_profile("production", PRODUCTION_LAYERS)
