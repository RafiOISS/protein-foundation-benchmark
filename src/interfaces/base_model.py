"""Base model interface for the Protein Foundation Model Benchmark Framework.

All protein model wrappers must inherit from BaseProteinModel.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..utils.logging import get_logger


logger = get_logger(__name__)


class BaseProteinModel(nn.Module, ABC):
    """Abstract base class for all protein foundation models.

    Provides a unified interface for forward pass, embedding extraction,
    and common utilities. All model implementations must subclass this.
    """

    def __init__(self, config: Dict[str, Any], device: str = "auto") -> None:
        super().__init__()
        self.config = config
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @abstractmethod
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            sequences: List of protein sequences.

        Returns:
            Model outputs (logits, hidden states, or embeddings).
        """
        pass

    @abstractmethod
    def extract_embeddings(
        self,
        sequences: List[str],
        layers: Union[str, List[int]] = "last",
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract per-sequence embeddings.

        Args:
            sequences: List of protein sequences.
            layers: Which layers to extract ('last', 'all', or list of indices).
            pooling: Pooling strategy ('mean', 'cls', 'max').

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        pass

    @abstractmethod
    def get_max_seq_len(self) -> int:
        """Return the maximum supported sequence length."""
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """Return the number of transformer / encoder layers."""
        pass

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def to(self, device: Union[str, torch.device]) -> "BaseProteinModel":
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(self.device)

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs) -> "BaseProteinModel":
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.get_embedding_dim()}, max_len={self.get_max_seq_len()})"