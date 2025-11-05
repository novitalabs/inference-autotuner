"""
Configuration layers and context definitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
from collections.abc import Mapping
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigLayer:
    """
    Configuration layer definition.

    A layer represents a piece of configuration that can be conditionally
    applied to build the final configuration object.
    """
    name: str
    data: dict | Callable[[TaskContext], dict]
    condition: Callable[[TaskContext], bool] | None = None

    def applies_to(self, ctx: TaskContext) -> bool:
        """Check if this layer should be applied to the given context."""
        if self.condition is None:
            return True
        try:
            return self.condition(ctx)
        except Exception:
            logger.debug(f"Layer {self.name} condition failed")
            return False

    def resolve(self, ctx: TaskContext) -> dict:
        """Resolve the layer data (call function if callable, otherwise return dict)."""
        payload = self.data(ctx) if callable(self.data) else self.data
        return copy.deepcopy(payload)


@dataclass
class TaskContext:
    """
    Task execution context containing all information needed for configuration decisions.
    """
    # Basic configuration
    model_name: str
    base_runtime: str                           # sglang, vllm, trtllm
    deployment_mode: str                        # docker, ome

    # Runtime parameters
    benchmark_task: str                         # text-to-text, etc.
    traffic_scenarios: list[str]
    num_concurrency: list[int]

    # Optimization objectives
    optimization_strategy: str                  # grid_search, bayesian
    optimization_objective: str                 # minimize_latency, etc.

    # SLO configuration (optional)
    slo_config: dict | None = None

    # Advanced options
    profiles: list[str] = field(default_factory=list)
    user_overrides: dict = field(default_factory=dict)
    override_mode: Literal["patch", "replace"] = "patch"

    # Environment configuration
    gpu_type: str | None = None
    total_gpus: int | None = None


def _deep_merge(target: dict, source: Mapping, *, allow_new: bool = True) -> dict:
    """
    Deep merge two dictionaries recursively.

    Args:
        target: The target dictionary to merge into (modified in place)
        source: The source dictionary to merge from
        allow_new: Whether to allow adding new keys not present in target

    Returns:
        The target dictionary (modified in place)
    """
    for key, value in source.items():
        if key not in target:
            if not allow_new:
                continue
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_merge(target[key], value, allow_new=allow_new)
        else:
            target[key] = copy.deepcopy(value)
    return target
