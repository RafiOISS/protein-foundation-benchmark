"""
Registry package for the Protein Foundation Model Benchmark Framework.

Registries provide a plug-and-play mechanism for models, datasets, and metrics.
"""

from .model_registry import ModelRegistry
from .dataset_registry import DatasetRegistry
from .metric_registry import MetricRegistry

__all__ = [
    "ModelRegistry",
    "DatasetRegistry",
    "MetricRegistry",
]