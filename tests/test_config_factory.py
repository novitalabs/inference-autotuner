"""
Unit tests for configuration factory.
"""

import pytest
from src.config.layers import ConfigLayer, TaskContext, _deep_merge
from src.config.factory import TaskConfigFactory
from src.config.profiles import register_builtin_profiles


def test_deep_merge_basic():
    """Test basic deep merge functionality."""
    target = {"a": 1, "b": {"c": 2}}
    source = {"b": {"d": 3}, "e": 4}
    result = _deep_merge(target, source)

    assert result["a"] == 1
    assert result["b"]["c"] == 2
    assert result["b"]["d"] == 3
    assert result["e"] == 4


def test_deep_merge_override():
    """Test that deep merge overrides values correctly."""
    target = {"a": 1, "b": {"c": 2}}
    source = {"a": 10, "b": {"c": 20}}
    result = _deep_merge(target, source)

    assert result["a"] == 10
    assert result["b"]["c"] == 20


def test_deep_merge_allow_new_false():
    """Test that allow_new=False prevents adding new keys."""
    target = {"a": 1}
    source = {"a": 10, "b": 2}
    result = _deep_merge(target, source, allow_new=False)

    assert result["a"] == 10
    assert "b" not in result


def test_config_layer_applies_to():
    """Test conditional application of config layers."""
    ctx = TaskContext(
        model_name="test-model",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency"
    )

    # Layer without condition should always apply
    layer = ConfigLayer("test", {"key": "value"})
    assert layer.applies_to(ctx)

    # Layer with True condition should apply
    layer_true = ConfigLayer("test", {"key": "value"}, condition=lambda ctx: True)
    assert layer_true.applies_to(ctx)

    # Layer with False condition should not apply
    layer_false = ConfigLayer("test", {"key": "value"}, condition=lambda ctx: False)
    assert not layer_false.applies_to(ctx)


def test_config_layer_resolve():
    """Test layer resolution with static and dynamic data."""
    ctx = TaskContext(
        model_name="test-model",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency"
    )

    # Static data
    layer_static = ConfigLayer("test", {"key": "value"})
    assert layer_static.resolve(ctx) == {"key": "value"}

    # Dynamic data (function)
    def dynamic_func(ctx: TaskContext) -> dict:
        return {"model": ctx.model_name}

    layer_dynamic = ConfigLayer("test", dynamic_func)
    assert layer_dynamic.resolve(ctx) == {"model": "test-model"}


def test_factory_create_basic():
    """Test basic configuration creation."""
    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency"
    )

    config, layers = TaskConfigFactory.create(ctx)

    # Check that base layers were applied
    assert "task_name" in config
    assert "llama-3-2-1b-instruct" in config["task_name"]
    assert config["base_runtime"] == "sglang"
    assert config["optimization"]["strategy"] == "grid_search"

    # Check that deployment mode layer was applied
    assert "runtime_image_tag" in config  # Docker-specific

    # Check that runtime layer was applied
    assert "schedule-policy" in config["parameters"]  # SGLang-specific

    # Verify applied layers
    assert "base-model" in layers
    assert "base-optimization" in layers
    assert "docker-defaults" in layers
    assert "sglang-defaults" in layers


def test_factory_with_profile():
    """Test configuration creation with profile."""
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
        profiles=["quick-test"]
    )

    config, layers = TaskConfigFactory.create(ctx)

    # Check that profile was applied
    assert config["optimization"]["max_iterations"] == 2  # From quick-test profile
    assert config["parameters"]["tp-size"] == [1]  # From quick-test profile

    # Verify profile layer was applied
    profile_layers = [l for l in layers if l.startswith("profile:quick-test:")]
    assert len(profile_layers) == 1


def test_factory_with_user_override():
    """Test configuration creation with user override."""
    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        user_overrides={
            "parameters": {
                "tp-size": [8]
            }
        }
    )

    config, layers = TaskConfigFactory.create(ctx)

    # Check that user override was applied
    assert config["parameters"]["tp-size"] == [8]

    # Verify user override layer was applied
    assert "user_patch" in layers


def test_factory_with_slo():
    """Test configuration creation with SLO constraints."""
    slo_config = {
        "ttft": {"threshold": 1.0, "weight": 2.0},
        "tpot": {"threshold": 0.05, "weight": 2.0}
    }

    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        slo_config=slo_config
    )

    config, layers = TaskConfigFactory.create(ctx)

    # Check that SLO was applied
    assert "slo" in config
    assert config["slo"]["ttft"]["threshold"] == 1.0
    assert config["slo"]["tpot"]["threshold"] == 0.05


def test_factory_with_total_gpus():
    """Test that total_gpus limits tp-size options."""
    ctx = TaskContext(
        model_name="llama-3-2-1b-instruct",
        base_runtime="sglang",
        deployment_mode="docker",
        benchmark_task="text-to-text",
        traffic_scenarios=["D(100,100)"],
        num_concurrency=[1, 4],
        optimization_strategy="grid_search",
        optimization_objective="minimize_latency",
        total_gpus=2
    )

    config, layers = TaskConfigFactory.create(ctx)

    # Check that tp-size was limited by total_gpus
    tp_sizes = config["parameters"]["tp-size"]
    assert all(tp <= 2 for tp in tp_sizes)
    assert 4 not in tp_sizes  # 4 > 2, should be filtered out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
