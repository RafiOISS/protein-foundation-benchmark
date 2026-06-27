"""CheckpointManager — centralized checkpoint management for ProteinBERT.

Responsibilities:
  - download pretrained weights into workspace checkpoint directory
  - verify checkpoint integrity (SHA-256 when available)
  - avoid duplicate downloads
  - load checkpoints
  - save checkpoints
  - resume interrupted training
  - maintain checkpoint metadata

All checkpoints remain inside checkpoints/proteinbert/.
No checkpoint written to default OS cache locations.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...utils.logging import get_logger


logger = get_logger(__name__)


CHECKPOINT_SUBDIR = "proteinbert"


@dataclass
class CheckpointMetadata:
    """Metadata for a single checkpoint."""
    epoch: int
    path: str
    timestamp: str = ""
    loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    optimizer_state: bool = False
    is_best: bool = False
    file_size_bytes: int = 0
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """Centralized checkpoint management.

    Usage:
        cm = CheckpointManager(workspace_root)
        cm.download_pretrained()
        cm.save(model, epoch=5, loss=0.1, val_loss=0.2)
        ckpt = cm.load_latest()
        best = cm.load_best()
    """

    def __init__(
        self,
        workspace_root: Union[str, Path],
        cache_manager: Optional[Any] = None,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        self._checkpoint_root = self._workspace_root / "checkpoints" / CHECKPOINT_SUBDIR
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._cache_manager = cache_manager
        self._manifest_path = self._checkpoint_root / "checkpoint_manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()
        logger.debug(f"CheckpointManager initialized (root={self._checkpoint_root})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def checkpoint_root(self) -> Path:
        return self._checkpoint_root

    @property
    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    # ------------------------------------------------------------------
    # Pretrained download
    # ------------------------------------------------------------------

    def download_pretrained(
        self,
        url: Optional[str] = None,
        checksum: Optional[str] = None,
    ) -> Path:
        """Download pretrained ProteinBERT weights.

        Uses CacheManager if available; otherwise downloads to
        checkpoints/proteinbert/pretrained/.

        Args:
            url: Download URL. Uses default if None.
            checksum: Expected SHA-256 checksum.

        Returns:
            Path to downloaded weights.
        """
        pretrained_dir = self._checkpoint_root / "pretrained"
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        # Avoid duplicate download via manifest
        if self._is_pretrained_downloaded(pretrained_dir):
            logger.info("Pretrained weights already downloaded, skipping")
            return self._find_pretrained_weights(pretrained_dir)

        if self._cache_manager is not None:
            cm = self._cache_manager
            if url and not cm.is_downloaded(url, checksum):
                self._do_download(url, pretrained_dir, checksum, cm)

        weight_path = self._find_pretrained_weights(pretrained_dir)
        logger.info(f"Pretrained weights ready at {weight_path}")
        return weight_path

    def _is_pretrained_downloaded(self, pretrained_dir: Path) -> bool:
        if not pretrained_dir.exists():
            return False
        files = list(pretrained_dir.iterdir())
        if not files:
            return False
        manifest_entry = self._manifest.get("pretrained_downloaded")
        if manifest_entry:
            return Path(manifest_entry).exists()
        return any(f.suffix in (".h5", ".hdf5", ".tf", ".index", ".ckpt") for f in files)

    def _find_pretrained_weights(self, pretrained_dir: Path) -> Path:
        for ext in (".h5", ".hdf5", ".tf", ".index"):
            matches = list(pretrained_dir.rglob(f"*{ext}"))
            if matches:
                return matches[0]
        return pretrained_dir

    def _do_download(
        self,
        url: str,
        dest_dir: Path,
        checksum: Optional[str],
        cache_manager: Any,
    ) -> Path:
        import urllib.request
        filename = url.split("/")[-1] or "pretrained_weights.h5"
        dest = dest_dir / filename

        logger.info(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)

        if checksum:
            actual = self._compute_checksum(dest)
            if actual != checksum:
                raise RuntimeError(
                    f"Checksum mismatch for {dest}: "
                    f"expected {checksum}, got {actual}"
                )

        cache_manager.record_download(url, dest, checksum)
        self._manifest["pretrained_downloaded"] = str(dest)
        self._save_manifest()
        return dest

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        epoch: int,
        loss: float = 0.0,
        val_loss: float = 0.0,
        accuracy: float = 0.0,
        val_accuracy: float = 0.0,
        optimizer: Optional[Any] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            model: TF model with save_weights().
            epoch: Current epoch number.
            loss: Training loss.
            val_loss: Validation loss.
            accuracy: Training accuracy.
            val_accuracy: Validation accuracy.
            optimizer: TF optimizer for state restoration.
            is_best: Whether this is the best checkpoint so far.

        Returns:
            Path to saved checkpoint directory.
        """
        epoch_dir = self._checkpoint_root / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        weights_path = str(epoch_dir / "weights.h5")
        model.save_weights(weights_path)
        logger.debug(f"Saved weights to {weights_path}")

        if optimizer is not None:
            try:
                optimizer_path = str(epoch_dir / "optimizer.npz")
                import numpy as np
                np.savez_optimizer(optimizer_path)
                weights = model.get_weights()
                np.savez(optimizer_path, *weights)

                opt_weights = optimizer.get_weights()
                import numpy as np
                opt_path = str(epoch_dir / "optimizer_weights.npz")
                np.savez(opt_path, *opt_weights)
            except Exception as e:
                logger.warning(f"Could not save optimizer state: {e}")

        if is_best:
            best_dir = self._checkpoint_root / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = str(best_dir / "weights.h5")
            model.save_weights(best_path)
            self._manifest["best_checkpoint"] = {
                "epoch": epoch,
                "path": best_path,
                "loss": loss,
                "val_loss": val_loss,
            }

        metadata = CheckpointMetadata(
            epoch=epoch,
            path=str(epoch_dir),
            loss=loss,
            val_loss=val_loss,
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            optimizer_state=optimizer is not None,
            is_best=is_best,
            file_size_bytes=self._dir_size(epoch_dir),
        )

        self._manifest.setdefault("checkpoints", [])
        self._manifest["checkpoints"].append(metadata.to_dict())
        self._manifest["latest_epoch"] = epoch
        self._save_manifest()

        logger.info(f"Checkpoint saved: epoch {epoch} ({loss:.4f} / {val_loss:.4f})")
        return epoch_dir

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Returns:
            Dict with checkpoint info, or None if no checkpoints exist.
        """
        ckpts = self._manifest.get("checkpoints", [])
        if not ckpts:
            return None
        latest = max(ckpts, key=lambda c: c["epoch"])
        return latest

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.

        Returns:
            Dict with best checkpoint info, or None.
        """
        return self._manifest.get("best_checkpoint")

    def get_latest_epoch(self) -> int:
        """Get the latest epoch number from manifest."""
        return self._manifest.get("latest_epoch", 0)

    def restore_model(
        self,
        model: Any,
        epoch: Optional[int] = None,
    ) -> int:
        """Restore model weights from a checkpoint.

        Args:
            model: TF model.
            epoch: Specific epoch to restore (latest if None).

        Returns:
            Restored epoch number (0 if no checkpoint).
        """
        if epoch is not None:
            ckpt_path = self._checkpoint_root / f"epoch_{epoch:04d}" / "weights.h5"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            latest = self.load_latest()
            if latest is None:
                logger.info("No checkpoint to restore")
                return 0
            epoch = latest["epoch"]
            ckpt_path = Path(latest["path"]) / "weights.h5"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint path missing: {ckpt_path}")
                return 0

        try:
            model.load_weights(str(ckpt_path))
            logger.info(f"Restored weights from epoch {epoch}")
        except Exception as e:
            logger.warning(f"Weight restore failed: {e}")
            return 0

        return epoch

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def get_resume_state(self) -> Dict[str, Any]:
        """Get state for resuming training.

        Returns:
            Dict with 'epoch' (int), 'checkpoints' (list), 'best' (dict or None).
        """
        return {
            "epoch": self.get_latest_epoch(),
            "checkpoints": self._manifest.get("checkpoints", []),
            "best": self._manifest.get("best_checkpoint"),
        }

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Corrupt manifest, starting fresh: {self._manifest_path}")
        return {"checkpoints": [], "latest_epoch": 0}

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)
        logger.debug(f"Checkpoint manifest saved ({len(self._manifest.get('checkpoints', []))} entries)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints."""
        return list(self._manifest.get("checkpoints", []))

    def info(self) -> Dict[str, Any]:
        """Return a summary of checkpoint state."""
        ckpts = self._manifest.get("checkpoints", [])
        return {
            "checkpoint_root": str(self._checkpoint_root),
            "total_checkpoints": len(ckpts),
            "latest_epoch": self.get_latest_epoch(),
            "has_best": self._manifest.get("best_checkpoint") is not None,
            "pretrained_downloaded": self._manifest.get("pretrained_downloaded") is not None,
        }

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(root={self._checkpoint_root}, "
            f"checkpoints={len(self._manifest.get('checkpoints', []))})"
        )
