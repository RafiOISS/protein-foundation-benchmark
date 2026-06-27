"""
Datasets package for the Protein Foundation Model Benchmark Framework.

Each dataset family is in its own subdirectory.
"""

from .base_dataset import BaseDataset, DatasetSplit, TaskType, DatasetInfo

__all__ = [
    "BaseDataset",
    "DatasetSplit",
    "TaskType",
    "DatasetInfo",
]