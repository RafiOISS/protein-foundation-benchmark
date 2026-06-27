"""Experiment lifecycle management.

Experiments track a single benchmark run across multiple model-dataset pairs.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger
from ..utils.io import save_json


logger = get_logger(__name__)


class ExperimentStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment:
    """Encapsulates a single experiment lifecycle."""

    def __init__(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        project_root: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.description = description
        self.tags = tags or []
        self.config = config or {}
        self.seed = seed
        self.status = ExperimentStatus.CREATED
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        self._project_root = Path(project_root)
        self._dir = self._project_root / "outputs" / "experiments" / f"{self.id}_{name}"
        self._dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment '{name}' created (id={self.id})")

    @property
    def dir(self) -> Path:
        return self._dir

    def start(self) -> None:
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        self._save_state()

    def complete(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        if results:
            self.results = results
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self._save_state()
        self._save_results()

    def fail(self, error: str) -> None:
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.config["error"] = error
        self._save_state()

    def add_result(self, result: Dict[str, Any]) -> None:
        self.results.append(result)

    def _save_state(self) -> None:
        save_json(self.to_dict(), self._dir / "experiment.json")

    def _save_results(self) -> None:
        save_json(self.results, self._dir / "results.json")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "config": self.config,
            "seed": self.seed,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "num_results": len(self.results),
        }

    def __repr__(self) -> str:
        return f"Experiment(name={self.name}, status={self.status.value}, id={self.id})"