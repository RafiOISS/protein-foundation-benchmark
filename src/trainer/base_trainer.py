"""Base trainer — abstract interface for all training implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        pass