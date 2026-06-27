"""
Protein Foundation Model Benchmark Framework

A production-quality benchmark framework for comparing pretrained
protein foundation models across multiple datasets and tasks.
"""

__version__ = "0.2.0"
__author__ = "RafiOISS"
__license__ = "MIT"

from .framework.benchmark import ProteinBenchmark
from .framework.experiment import Experiment, ExperimentStatus

# Import datasets to trigger registration with DatasetRegistry
from . import datasets  # noqa: F401

__all__ = [
    "ProteinBenchmark",
    "Experiment",
    "ExperimentStatus",
    "__version__",
]