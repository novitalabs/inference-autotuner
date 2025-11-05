"""
Layered configuration factory for building task configurations.
"""

from typing import ClassVar
import copy
import logging
from .layers import ConfigLayer, TaskContext, _deep_merge

logger = logging.getLogger(__name__)


class TaskConfigFactory:
    """
    Layered configuration factory that builds task configurations by applying
    multiple configuration layers in sequence.

    Configuration layers are applied in this order:
    1. Base layers (model, optimization)
    2. Deployment mode layers (docker, ome)
    3. Runtime layers (sglang, vllm, trtllm)
    4. User-defined profiles (registered presets)
    5. User overrides (patch or replace mode)
    6. Finalization (apply constraints and defaults)
    """

    PROFILE_REGISTRY: ClassVar[dict[str, list[ConfigLayer]]] = {}

    @classmethod
    def register_profile(cls, name: str, layers: list[ConfigLayer]) -> None:
        """Register a configuration preset profile."""
        cls.PROFILE_REGISTRY[name] = layers
        logger.info(f"Registered profile: {name}")

    @classmethod
    def create(cls, ctx: TaskContext) -> tuple[dict, list[str]]:
        """
        Create configuration from context.

        Args:
            ctx: Task context containing all configuration parameters

        Returns:
            (config_dict, applied_layers): Configuration dict and list of applied layer names
        """
        config_dict: dict = {}
        applied_layers: list[str] = []

        # 1. Apply base layers
        for layer in cls._base_layers():
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 2. Apply deployment mode layers
        for layer in cls._deployment_mode_layers(ctx):
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 3. Apply runtime layers
        for layer in cls._runtime_layers(ctx):
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        # 4. Apply user presets
        for profile in ctx.profiles:
            layers = cls.PROFILE_REGISTRY.get(profile)
            if not layers:
                logger.warning(f"Profile '{profile}' not found, skipping")
                continue
            for layer in layers:
                if layer.applies_to(ctx):
                    _deep_merge(config_dict, layer.resolve(ctx))
                    applied_layers.append(f"profile:{profile}:{layer.name}")

        # 5. Apply user overrides
        if ctx.user_overrides:
            if ctx.override_mode == "replace":
                config_dict = copy.deepcopy(ctx.user_overrides)
                applied_layers.append("user_replace")
            else:
                _deep_merge(config_dict, ctx.user_overrides, allow_new=True)
                applied_layers.append("user_patch")

        # 6. Finalization
        cls._finalize(config_dict, ctx)

        logger.info(f"Applied layers: {applied_layers}")
        return config_dict, applied_layers

    @classmethod
    def _base_layers(cls) -> list[ConfigLayer]:
        """Base configuration layers."""
        return [
            ConfigLayer("base-model", cls._base_model_layer),
            ConfigLayer("base-optimization", cls._base_optimization_layer),
        ]

    @classmethod
    def _deployment_mode_layers(cls, ctx: TaskContext) -> list[ConfigLayer]:
        """Deployment mode specific layers."""
        if ctx.deployment_mode == "docker":
            return [ConfigLayer("docker-defaults", cls._docker_defaults_layer)]
        elif ctx.deployment_mode == "ome":
            return [ConfigLayer("ome-defaults", cls._ome_defaults_layer)]
        return []

    @classmethod
    def _runtime_layers(cls, ctx: TaskContext) -> list[ConfigLayer]:
        """Runtime engine specific layers."""
        if ctx.base_runtime == "sglang":
            return [ConfigLayer("sglang-defaults", cls._sglang_defaults_layer)]
        elif ctx.base_runtime == "vllm":
            return [ConfigLayer("vllm-defaults", cls._vllm_defaults_layer)]
        elif ctx.base_runtime == "trtllm":
            return [ConfigLayer("trtllm-defaults", cls._trtllm_defaults_layer)]
        return []

    @staticmethod
    def _base_model_layer(ctx: TaskContext) -> dict:
        """Base model configuration."""
        return {
            "task_name": f"{ctx.model_name}_{ctx.optimization_strategy}",
            "model": {
                "id_or_path": ctx.model_name,
                "namespace": "autotuner"
            },
            "base_runtime": ctx.base_runtime,
        }

    @staticmethod
    def _base_optimization_layer(ctx: TaskContext) -> dict:
        """Base optimization configuration."""
        return {
            "optimization": {
                "strategy": ctx.optimization_strategy,
                "objective": ctx.optimization_objective,
                "max_iterations": 10,
                "timeout_per_iteration": 600
            },
            "benchmark": {
                "task": ctx.benchmark_task,
                "model_name": ctx.model_name,
                "traffic_scenarios": ctx.traffic_scenarios,
                "num_concurrency": ctx.num_concurrency,
            }
        }

    @staticmethod
    def _docker_defaults_layer(ctx: TaskContext) -> dict:
        """Docker mode default configuration."""
        return {
            "runtime_image_tag": "v0.5.2-cu126",
            "parameters": {
                "tp-size": [1, 2, 4],
                "mem-fraction-static": [0.85, 0.9]
            }
        }

    @staticmethod
    def _ome_defaults_layer(ctx: TaskContext) -> dict:
        """OME mode default configuration."""
        return {
            "parameters": {
                "tp-size": [1, 2, 4, 8],
                "mem-fraction-static": [0.8, 0.85, 0.9]
            }
        }

    @staticmethod
    def _sglang_defaults_layer(ctx: TaskContext) -> dict:
        """SGLang runtime default parameters."""
        return {
            "parameters": {
                "schedule-policy": ["lpm"],
                "enable-torch-compile": [True, False]
            }
        }

    @staticmethod
    def _vllm_defaults_layer(ctx: TaskContext) -> dict:
        """vLLM runtime default parameters."""
        return {
            "parameters": {
                "enable-chunked-prefill": [True, False]
            }
        }

    @staticmethod
    def _trtllm_defaults_layer(ctx: TaskContext) -> dict:
        """TensorRT-LLM runtime default parameters."""
        return {
            "parameters": {
                "max-batch-size": [32, 64, 128]
            }
        }

    @classmethod
    def _finalize(cls, config: dict, ctx: TaskContext) -> None:
        """Finalize configuration by applying constraints and defaults."""
        # Apply SLO configuration
        if ctx.slo_config:
            config["slo"] = ctx.slo_config

        # Limit GPU count if specified
        if ctx.total_gpus and "parameters" in config:
            if "tp-size" in config["parameters"]:
                config["parameters"]["tp-size"] = [
                    tp for tp in config["parameters"]["tp-size"]
                    if tp <= ctx.total_gpus
                ]
