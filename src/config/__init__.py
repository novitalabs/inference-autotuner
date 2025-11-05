"""
Configuration management using layered config factory pattern.
"""

from .layers import ConfigLayer, TaskContext, _deep_merge
from .factory import TaskConfigFactory

__all__ = [
    "ConfigLayer",
    "TaskContext",
    "TaskConfigFactory",
    "_deep_merge",
]
