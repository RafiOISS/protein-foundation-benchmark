"""
Models package for the Protein Foundation Model Benchmark Framework.

Each model family is in its own subdirectory for clean separation.
"""

from .esm2 import ESM2
from .protbert import ProtBERT
from .prott5 import ProtT5
from .proteinbert import ProteinBERT
from .cnn import CNNBaseline
from .bilstm import BiLSTMBaseline

__all__ = [
    "ESM2",
    "ProtBERT",
    "ProtT5",
    "ProteinBERT",
    "CNNBaseline",
    "BiLSTMBaseline",
]