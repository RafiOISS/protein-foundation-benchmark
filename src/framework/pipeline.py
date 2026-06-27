"""Pipeline — orchestrates a single model + dataset run.

Handles data loading, embedding extraction, training, evaluation, and reporting.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging import get_logger
from ..framework.checkpoint import Checkpoint
from ..framework.experiment import Experiment
from ..interfaces.base_dataset import BaseDataset, DatasetSplit
from ..interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class PipelineStage(Enum):
    SETUP = "setup"
    DATA = "data"
    EMBEDDING = "embedding"
    TRAINING = "training"
    EVALUATION = "evaluation"
    REPORT = "report"
    COMPLETE = "complete"


@dataclass
class PipelineConfig:
    model_name: str
    dataset_name: str
    device: str = "auto"
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    max_seq_len: int = 1022
    early_stopping_patience: int = 5


class Pipeline:
    """Orchestrates a single model + dataset run.

    Sequences: data loading, embedding extraction (optional),
    training (optional), evaluation, and result collection.
    """

    def __init__(
        self,
        model: BaseProteinModel,
        experiment: Experiment,
        dataset_name: str,
        model_name: str,
        config: Optional[PipelineConfig] = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.experiment = experiment
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.config = config or PipelineConfig(model_name=model_name, dataset_name=dataset_name)
        self.device = device

        self._stage = PipelineStage.SETUP
        self._results: Dict[str, Any] = {}
        self._dir = experiment.dir / f"{model_name}_{dataset_name}"
        self._dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = Checkpoint(directory=self._dir / "checkpoints")

    def _make_loader(self, dataset: BaseDataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
        )

    def run(
        self,
        train_dataset: Optional[BaseDataset] = None,
        val_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline."""
        start_time = time.time()
        self._stage = PipelineStage.DATA

        # Dataloaders
        loaders = {}
        if train_dataset:
            loaders["train"] = self._make_loader(train_dataset, shuffle=True)
        if val_dataset:
            loaders["val"] = self._make_loader(val_dataset)
        if test_dataset:
            loaders["test"] = self._make_loader(test_dataset)

        # Embedding extraction
        if test_dataset:
            self._stage = PipelineStage.EMBEDDING
            embeddings = self._extract_embeddings(loaders["test"])
            self._results["embeddings_shape"] = list(embeddings.shape)

        # Training
        if train_dataset and val_dataset:
            self._stage = PipelineStage.TRAINING
            history = self._train(loaders["train"], loaders["val"])
            self._results["history"] = history

        # Evaluation
        if test_dataset:
            self._stage = PipelineStage.EVALUATION
            metrics = self._evaluate(loaders["test"])
            self._results["metrics"] = metrics

        duration = time.time() - start_time
        self._results["duration"] = duration
        self._results["model_name"] = self.model_name
        self._results["dataset_name"] = self.dataset_name

        self._stage = PipelineStage.COMPLETE
        logger.info(f"Pipeline complete ({duration:.2f}s)")

        return self._results

    def _extract_embeddings(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        all_emb = []
        with torch.no_grad():
            for batch in loader:
                emb = self.model.extract_embeddings(batch["sequence"])
                all_emb.append(emb.cpu())
        embeddings = torch.cat(all_emb)
        torch.save(embeddings, self._dir / "embeddings.pt")
        return embeddings

    def _train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience = self.config.early_stopping_patience
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch["sequence"])
                targets = batch["target"].to(self.model.device)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(batch["sequence"])
                    targets = batch["target"].to(self.model.device)
                    val_loss += loss_fn(outputs, targets).item()
            val_loss /= len(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.checkpoint.save(self.model, epoch=epoch, step=epoch * len(train_loader), metrics={"val_loss": val_loss})

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["sequence"])
                all_preds.append(outputs.cpu())
                all_targets.append(batch["target"].cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        # Basic metrics
        mse = nn.MSELoss()(preds, targets).item()
        mae = nn.L1Loss()(preds, targets).item()

        metrics = {"mse": mse, "mae": mae}

        torch.save({"predictions": preds, "targets": targets}, self._dir / "predictions.pt")
        return metrics