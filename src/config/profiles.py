"""
Built-in configuration profiles for common use cases.
"""

from .layers import ConfigLayer
from .factory import TaskConfigFactory, ProfileMetadata


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

HIGH_THROUGHPUT_METADATA = ProfileMetadata(
    name="high-throughput",
    description="Maximize request throughput with high concurrency testing",
    use_case="Batch processing workloads, high-traffic scenarios",
    tags=["throughput", "batch", "high-concurrency"],
    recommended_for=["production", "multi-gpu", "high-traffic"]
)

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

LOW_LATENCY_METADATA = ProfileMetadata(
    name="low-latency",
    description="Minimize response latency for real-time interactive workloads",
    use_case="Chatbots, real-time applications, interactive services",
    tags=["latency", "real-time", "interactive"],
    recommended_for=["production", "interactive", "single-gpu"]
)

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

QUICK_TEST_METADATA = ProfileMetadata(
    name="quick-test",
    description="Fast validation with minimal experiments for development and testing",
    use_case="Development, debugging, smoke testing",
    tags=["testing", "development", "quick"],
    recommended_for=["development", "debugging", "ci-cd"]
)

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

PRODUCTION_METADATA = ProfileMetadata(
    name="production",
    description="Comprehensive testing with SLO constraints for production deployment",
    use_case="Production readiness testing with strict performance requirements",
    tags=["production", "slo", "comprehensive"],
    recommended_for=["production", "deployment", "slo-critical"]
)

# Cost-optimization preset (balance performance with minimal resources)
COST_OPTIMIZATION_LAYERS = [
    ConfigLayer(
        name="cost-optimization-params",
        data={
            "optimization": {
                "objective": "minimize_latency",  # Optimize for efficiency
                "max_iterations": 12
            },
            "parameters": {
                "mem-fraction-static": [0.75, 0.85],  # Lower memory usage
                "tp-size": [1, 2],  # Prefer fewer GPUs
            },
            "benchmark": {
                "num_concurrency": [4, 8, 16]  # Moderate concurrency
            }
        }
    )
]

COST_OPTIMIZATION_METADATA = ProfileMetadata(
    name="cost-optimization",
    description="Balance performance with minimal resource usage and cost",
    use_case="Budget-conscious deployments, development environments",
    tags=["cost", "efficiency", "budget"],
    recommended_for=["development", "startup", "single-gpu"]
)

# Balanced preset (good starting point for most workloads)
BALANCED_LAYERS = [
    ConfigLayer(
        name="balanced-params",
        data={
            "optimization": {
                "max_iterations": 15,
                "timeout_per_iteration": 600
            },
            "parameters": {
                "mem-fraction-static": [0.8, 0.85, 0.9],
                "tp-size": [1, 2, 4],
            },
            "benchmark": {
                "num_concurrency": [4, 8, 16]
            }
        }
    )
]

BALANCED_METADATA = ProfileMetadata(
    name="balanced",
    description="Balanced configuration exploring common parameter ranges",
    use_case="General purpose testing, exploring parameter space",
    tags=["balanced", "general", "exploration"],
    recommended_for=["general", "exploration", "baseline"]
)


def register_builtin_profiles():
    """Register all built-in configuration profiles with metadata."""
    TaskConfigFactory.register_profile("high-throughput", HIGH_THROUGHPUT_LAYERS, HIGH_THROUGHPUT_METADATA)
    TaskConfigFactory.register_profile("low-latency", LOW_LATENCY_LAYERS, LOW_LATENCY_METADATA)
    TaskConfigFactory.register_profile("quick-test", QUICK_TEST_LAYERS, QUICK_TEST_METADATA)
    TaskConfigFactory.register_profile("production", PRODUCTION_LAYERS, PRODUCTION_METADATA)
    TaskConfigFactory.register_profile("cost-optimization", COST_OPTIMIZATION_LAYERS, COST_OPTIMIZATION_METADATA)
    TaskConfigFactory.register_profile("balanced", BALANCED_LAYERS, BALANCED_METADATA)


