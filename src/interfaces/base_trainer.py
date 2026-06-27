"""Base trainer interface for the Protein Foundation Model Benchmark Framework.

All training implementations must inherit from BaseTrainer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Defines the interface for model training, validation, and checkpointing.
    """

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run training loop.

        Args:
            model: Model to train.
            train_loader: Training data.
            val_loader: Validation data.
            **kwargs: Additional training parameters.

        Returns:
            Training history dictionary.
        """
        pass

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, float]:
        """Run validation loop.

        Args:
            model: Model to evaluate.
            val_loader: Validation data.
            **kwargs: Additional validation parameters.

        Returns:
            Dictionary of validation metrics.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path], **kwargs) -> None:
        """Save trainer state."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load trainer state."""
        pass