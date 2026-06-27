"""
TAPE (Tasks Assessing Protein Embeddings) dataset implementations.

Includes: fluorescence, stability, PPI, secondary structure, remote homology.
"""

from ..base_dataset import BaseDataset, DatasetSplit, TaskType, DatasetInfo

__all__ = []  # Register datasets via DatasetRegistry