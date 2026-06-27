"""Datasets package for the Protein Foundation Model Benchmark Framework.

Registers all built-in datasets with the DatasetRegistry at import time.
"""

from ..registry.dataset_registry import DatasetRegistry
from ..utils.logging import get_logger

# Import datasets to trigger registration
from .tape_ss3 import TapeSS3Dataset


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Dataset Registration
# ------------------------------------------------------------------

DatasetRegistry.register("tape_ss3", TapeSS3Dataset)

logger.debug("Registered built-in dataset: tape_ss3")


__all__ = [
    "TapeSS3Dataset",
]
