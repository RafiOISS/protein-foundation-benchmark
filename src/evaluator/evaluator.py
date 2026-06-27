"""Evaluator — evaluates trained models on benchmark datasets."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..metrics.metrics import compute_metrics
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Evaluator:
    """Evaluates models on datasets with comprehensive metrics."""

    def __init__(self, device: str = "auto") -> None:
        self.device = torch.device(device) if device != "auto" else None

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: Optional[List[str]] = None,
        task_type: str = "regression",
        return_predictions: bool = True,
    ) -> Dict[str, Any]:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(batch["sequence"])
                all_preds.append(outputs.cpu())
                all_targets.append(batch["target"].cpu())

        predictions = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        metrics_dict = compute_metrics(predictions, targets, metrics or [], task_type)

        result = {"metrics": metrics_dict}
        if return_predictions:
            result["predictions"] = predictions
            result["targets"] = targets

        return result

    def evaluate_splits(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        metrics: Optional[List[str]] = None,
        task_type: str = "regression",
    ) -> Dict[str, Dict[str, Any]]:
        return {
            name: self.evaluate(model, loader, metrics, task_type)
            for name, loader in loaders.items()
        }