"""
Configuration management using layered config factory pattern.
"""

from .layers import ConfigLayer, TaskContext, _deep_merge
from .factory import TaskConfigFactory, ProfileMetadata

__all__ = [
    "ConfigLayer",
    "TaskContext",
    "TaskConfigFactory",
    "ProfileMetadata",
    "_deep_merge",
]
