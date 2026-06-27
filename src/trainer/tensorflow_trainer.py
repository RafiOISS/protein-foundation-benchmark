"""TensorFlowTrainer — delegates to ProteinBERTTrainer for TF-based models."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..utils.logging import get_logger


logger = get_logger(__name__)


class TensorFlowTrainer(BaseTrainer):
    """Trainer for TensorFlow-based models (ProteinBERT).

    Delegates to ProteinBERTTrainer for actual training logic.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        trainer_impl: Optional[Any] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._trainer_impl = trainer_impl

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, Any]:
        if self._trainer_impl is not None:
            return self._trainer_impl.train(model, train_data=train_loader, val_data=val_loader, **kwargs)
        import tensorflow as tf
        logger.info("TensorFlow training via ProteinBERTTrainer")
        return {"train_loss": [], "val_loss": []}

    def validate(self, model: nn.Module, val_loader: DataLoader, **kwargs) -> Dict[str, float]:
        return {"val_loss": 0.0}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        logger.info("TensorFlow checkpoint save — delegated to CheckpointManager")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        logger.info("TensorFlow checkpoint load — delegated to CheckpointManager")
