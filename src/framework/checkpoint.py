"""Checkpoint — handles model checkpointing with resume support."""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class CheckpointRecord:
    path: Path
    epoch: int
    step: int
    metrics: Dict[str, float]
    timestamp: str
    is_best: bool = False


class Checkpoint:
    """Manages model checkpoints with best / last tracking and cleanup."""

    def __init__(
        self,
        directory: Union[str, Path],
        max_keep: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
        filename_pattern: str = "epoch={epoch:03d}-{monitor:.4f}",
    ) -> None:
        self.directory = Path(directory)
        self.max_keep = max_keep
        self.monitor = monitor
        self.mode = mode
        self.filename_pattern = filename_pattern

        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: List[CheckpointRecord] = []
        self._best: Optional[CheckpointRecord] = None
        self._best_value: Optional[float] = None

    def save(
        self,
        model: nn.Module,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Path:
        metrics = metrics or {}
        value = metrics.get(self.monitor, 0.0)
        is_best = self._is_better(value)

        filename = self.filename_pattern.format(epoch=epoch, monitor=value)
        path = self.directory / f"{filename}.pt"

        state = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, path)

        record = CheckpointRecord(path=path, epoch=epoch, step=step, metrics=metrics, timestamp=state["timestamp"], is_best=is_best)
        self._records.append(record)

        if is_best:
            self._best = record
            self._best_value = value
            best_path = self.directory / "best.pt"
            shutil.copy2(path, best_path)

        last_path = self.directory / "last.pt"
        shutil.copy2(path, last_path)

        self._cleanup()
        return path

    def _is_better(self, value: float) -> bool:
        if self._best_value is None:
            return True
        return value < self._best_value if self.mode == "min" else value > self._best_value

    def _cleanup(self) -> None:
        if len(self._records) <= self.max_keep:
            return
        self._records.sort(key=lambda r: r.timestamp)
        for r in self._records[: -self.max_keep]:
            if not r.is_best and r.path.exists():
                r.path.unlink()
        self._records = self._records[-self.max_keep :]

    def load(
        self,
        model: nn.Module,
        path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        load_last: bool = False,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        if load_best:
            path = self.directory / "best.pt"
        elif load_last:
            path = self.directory / "last.pt"
        elif path is None:
            if self._best:
                path = self._best.path
            elif self._records:
                path = self._records[-1].path
            else:
                raise FileNotFoundError("No checkpoint available")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        device = device or torch.device("cpu")
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

        logger.info(f"Loaded checkpoint: {path} (epoch={state['epoch']})")
        return state

    @property
    def best(self) -> Optional[CheckpointRecord]:
        return self._best

    @property
    def latest(self) -> Optional[CheckpointRecord]:
        return self._records[-1] if self._records else None

    def list(self) -> List[CheckpointRecord]:
        return self._records.copy()