"""TorchTrainer — PyTorch training loop for ESM2, ProtBERT, ProtT5, CNN, BiLSTM."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..framework.checkpoint import Checkpoint
from ..utils.logging import get_logger


logger = get_logger(__name__)


class TorchTrainer(BaseTrainer):
    """PyTorch trainer with checkpointing, logging, and early stopping."""

    def __init__(
        self,
        checkpoint: Optional[Checkpoint] = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint = checkpoint
        self._device = torch.device(device) if device != "auto" else None
        self._history: Dict[str, List[float]] = {}

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch["sequence"])
                targets = batch["target"].to(model.device)
                loss = loss_fn(outputs, targets)
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch["sequence"])
                    targets = batch["target"].to(model.device)
                    val_loss += loss_fn(outputs, targets).item()
            val_loss /= len(val_loader)

            if scheduler:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.checkpoint:
                self.checkpoint.save(model, epoch=epoch, step=epoch * len(train_loader), metrics={"val_loss": val_loss}, optimizer=optimizer, scheduler=scheduler)

            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self._history = history
        return history

    def validate(self, model: nn.Module, val_loader: DataLoader, **kwargs) -> Dict[str, float]:
        model.eval()
        loss_fn = nn.MSELoss()
        total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["sequence"])
                targets = batch["target"].to(model.device)
                total += loss_fn(outputs, targets).item()
        return {"val_loss": total / len(val_loader)}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        torch.save(self._history, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        self._history = torch.load(path)