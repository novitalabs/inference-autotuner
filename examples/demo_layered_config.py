#!/usr/bin/env python3
"""
Demo script showcasing the Layered Config Factory feature.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.layers import TaskContext
from config.factory import TaskConfigFactory
from config.profiles import register_builtin_profiles


def demo_basic_config():
    """Demonstrate basic configuration creation."""
    print("=" * 80)
    print("Demo 1: Basic Configuration")
    print("=" * 80)

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4, 8],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency"
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print("Generated configuration:")
    print(json.dumps(config, indent=2))


def demo_quick_test_profile():
    """Demonstrate using quick-test profile."""
    print("\n" + "=" * 80)
    print("Demo 2: Quick Test Profile")
    print("=" * 80)

    register_builtin_profiles()

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="vllm",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        profiles=["quick-test"]  # Use quick-test profile
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print("Note: max_iterations=2 and tp-size=[1] from quick-test profile")
    print(f"max_iterations: {config['optimization']['max_iterations']}")
    print(f"tp-size: {config['parameters']['tp-size']}")


def demo_production_profile():
    """Demonstrate using production profile with SLO."""
    print("\n" + "=" * 80)
    print("Demo 3: Production Profile (with SLO)")
    print("=" * 80)

    register_builtin_profiles()

    ctx = TaskContext(
        model_name="llama-3-1-70b",
        base_runtime="sglang",
        deployment_mode="ome",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)", "D(200,200)"],
        num_concurrency=[1, 4, 8, 16],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        profiles=["production"]  # Use production profile
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print("SLO configuration from production profile:")
    print(json.dumps(config.get("slo", {}), indent=2))
    print(f"\nmax_iterations: {config['optimization']['max_iterations']}")


def demo_multiple_profiles():
    """Demonstrate combining multiple profiles."""
    print("\n" + "=" * 80)
    print("Demo 4: Combining Multiple Profiles")
    print("=" * 80)

    register_builtin_profiles()

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        profiles=["low-latency", "production"]  # Combine two profiles
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print("Configuration combines low-latency params + production SLO:")
    print(f"  tp-size (from low-latency): {config['parameters']['tp-size']}")
    print(f"  num_concurrency (from low-latency): {config['benchmark']['num_concurrency']}")
    print(f"  max_iterations (from production): {config['optimization']['max_iterations']}")
    print(f"  SLO (from production): {list(config.get('slo', {}).keys())}")


def demo_user_override():
    """Demonstrate user overrides."""
    print("\n" + "=" * 80)
    print("Demo 5: User Overrides")
    print("=" * 80)

    register_builtin_profiles()

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        profiles=["quick-test"],
        user_overrides={
            "parameters": {
                "tp-size": [2, 4]  # Override quick-test's tp-size=[1]
            },
            "optimization": {
                "max_iterations": 5  # Override quick-test's max_iterations=2
            }
        }
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print("User overrides applied (patch mode):")
    print(f"  tp-size: {config['parameters']['tp-size']} (overridden from [1])")
    print(f"  max_iterations: {config['optimization']['max_iterations']} (overridden from 2)")


def demo_gpu_constraint():
    """Demonstrate GPU constraints."""
    print("\n" + "=" * 80)
    print("Demo 6: GPU Constraints")
    print("=" * 80)

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        total_gpus=2  # Limit to 2 GPUs
    )

    config, layers = TaskConfigFactory.create(ctx)

    print(f"\nApplied layers: {layers}\n")
    print(f"With total_gpus=2, tp-size options are filtered:")
    print(f"  Original docker defaults: [1, 2, 4]")
    print(f"  After filtering: {config['parameters']['tp-size']}")


def main():
    """Run all demos."""
    demo_basic_config()
    demo_quick_test_profile()
    demo_production_profile()
    demo_multiple_profiles()
    demo_user_override()
    demo_gpu_constraint()

    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
