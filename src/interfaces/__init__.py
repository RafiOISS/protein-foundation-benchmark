"""
Interfaces package for the Protein Foundation Model Benchmark Framework.

Defines abstract base classes that all components must implement.
"""

from .base_model import BaseProteinModel
from .base_dataset import BaseDataset, DatasetSplit, TaskType
from .base_trainer import BaseTrainer
from .base_reporter import BaseReporter
from .base_metric import BaseMetric

__all__ = [
    "BaseProteinModel",
    "BaseDataset",
    "DatasetSplit",
    "TaskType",
    "BaseTrainer",
    "BaseReporter",
    "BaseMetric",
]