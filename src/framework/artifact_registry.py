"""ArtifactRegistry — tracks datasets, embeddings, predictions, figures, and metrics.

Each artifact carries a checksum for integrity verification.
"""

import hashlib
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger
from ..utils.io import save_json, save_yaml


logger = get_logger(__name__)


class ArtifactType(Enum):
    DATASET = "dataset"
    EMBEDDING = "embedding"
    PREDICTION = "prediction"
    FIGURE = "figure"
    METRIC = "metric"
    MODEL = "model"
    CONFIG = "config"
    REPORT = "report"


@dataclass
class Artifact:
    name: str
    type: ArtifactType
    path: Path
    checksum: str
    size_bytes: int
    created_at: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArtifactRegistry:
    """Tracks all experiment artifacts."""

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, Artifact] = {}
        self._load()

    def _checksum(self, path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def track(
        self,
        name: str,
        file_path: Union[str, Path],
        type: ArtifactType,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        copy: bool = True,
    ) -> Artifact:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        target = self.directory / type.value / file_path.name
        if copy:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
        else:
            target = file_path

        artifact = Artifact(
            name=name,
            type=type,
            path=target,
            checksum=self._checksum(target),
            size_bytes=target.stat().st_size,
            created_at=datetime.now().isoformat(),
            tags=tags or [],
            metadata=metadata or {},
        )

        self._artifacts[name] = artifact
        self._save()

        logger.info(f"Tracked artifact: {name} ({type.value}, {target})")
        return artifact

    def get(self, name: str) -> Optional[Artifact]:
        return self._artifacts.get(name)

    def list(self, type: Optional[ArtifactType] = None) -> List[Artifact]:
        if type:
            return [a for a in self._artifacts.values() if a.type == type]
        return list(self._artifacts.values())

    def verify(self) -> Dict[str, bool]:
        results = {}
        for name, art in self._artifacts.items():
            ok = art.path.exists() and self._checksum(art.path) == art.checksum
            results[name] = ok
            if not ok:
                logger.warning(f"Artifact verification failed: {name}")
        return results

    def _save(self) -> None:
        data = {}
        for name, art in self._artifacts.items():
            d = asdict(art)
            d["type"] = art.type.value
            d["path"] = str(art.path)
            data[name] = d
        save_json(data, self.directory / "artifacts.json")

    def _load(self) -> None:
        path = self.directory / "artifacts.json"
        if not path.exists():
            return
        import json
        with open(path) as f:
            data = json.load(f)
        for name, d in data.items():
            d["path"] = Path(d["path"])
            d["type"] = ArtifactType(d["type"])
            self._artifacts[name] = Artifact(**d)