"""Base dataset interface for the Protein Foundation Model Benchmark Framework.

All benchmark datasets must inherit from BaseDataset.
"""

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import ensure_dir, save_csv, save_json
from ..utils.logging import get_logger


logger = get_logger(__name__)


class TaskType(Enum):
    """Supported task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    TOKEN_CLASSIFICATION = "token_classification"


class DatasetSplit(Enum):
    """Standard dataset splits."""
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


@dataclass
class DatasetInfo:
    """Metadata describing a dataset."""
    name: str
    task_type: TaskType
    num_classes: Optional[int] = None
    num_samples: Optional[Dict[str, int]] = None
    sequence_length: Optional[Dict[str, float]] = None
    metrics: List[str] = None
    description: str = ""


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_framework_version() -> str:
    """Get framework version string."""
    try:
        from .. import __version__
        return __version__
    except ImportError:
        return "unknown"


class BaseDataset(Dataset, ABC):
    """Abstract base class for protein sequence datasets.

    All benchmark datasets must inherit from this class and implement
    the abstract methods.
    """

    # Subclasses set these
    DATASET_NAME: str = ""
    DATASET_VERSION: str = "1.0.0"
    PREPROCESSING_VERSION: str = "1.0.0"
    DOWNLOAD_URL: Optional[str] = None
    EXPECTED_SHA256: Optional[str] = None
    REQUIRED_FILES: Dict[str, Optional[str]] = {}

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_seq_len: int = 1022,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.config = config or {}

        self._sequences: List[str] = []
        self._targets: List[Any] = []
        self._metadata: List[Dict[str, Any]] = []

        # Cache directories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.statistics_dir = self.data_dir / "statistics"
        self.figures_dir = self.data_dir / "figures"
        self.metadata_dir = self.data_dir / "metadata"

        self._load_data()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def download(self, force: bool = False) -> Path:
        """Download the dataset.

        Skips if raw files already exist and pass verification.
        Subclasses should override DOWNLOAD_URL or this method.

        Args:
            force: Re-download even if files exist.

        Returns:
            Path to raw data directory.
        """
        if not force and self._is_downloaded():
            logger.info(f"{self.DATASET_NAME}: raw data already downloaded")
            return self.raw_dir

        ensure_dir(self.raw_dir)

        if self.DOWNLOAD_URL:
            from ..datasets.downloader import download_and_extract
            download_and_extract(
                url=self.DOWNLOAD_URL,
                dest_dir=self.raw_dir,
                expected_sha256=self.EXPECTED_SHA256,
            )
        else:
            raise NotImplementedError(
                f"{type(self).__name__} must override download() or set DOWNLOAD_URL"
            )

        logger.info(f"{self.DATASET_NAME}: download complete -> {self.raw_dir}")
        return self.raw_dir

    def verify(self) -> Dict[str, Any]:
        """Verify dataset integrity.

        Checks: required files exist, split counts, sequence validity,
        label validity, duplicates, missing values.

        Returns:
            Dict with verification results.
        """
        results: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        # Check required files
        if self.REQUIRED_FILES:
            from ..datasets.downloader import list_missing_files
            missing = list_missing_files(self.raw_dir, self.REQUIRED_FILES)
            if missing:
                for path, reason in missing:
                    results["errors"].append(f"Required file missing/invalid: {path} ({reason})")
                results["valid"] = False
            results["checks"]["required_files"] = len(missing) == 0

        # Verify sequences contain only valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        invalid_count = 0
        for seq in self._sequences:
            invalid_aas = set(seq.upper()) - valid_aas
            if invalid_aas - {"X", "B", "Z", "J", "U", "O"}:
                invalid_count += 1
        if invalid_count:
            results["warnings"].append(f"{invalid_count} sequences contain non-standard amino acids")
        results["checks"]["sequence_validity"] = invalid_count == 0

        # Check split counts
        results["checks"]["split_loaded"] = len(self._sequences) > 0
        if len(self._sequences) == 0:
            results["errors"].append(f"Split '{self.split.value}' has 0 sequences")

        # Check label validity
        if self._targets:
            invalid_labels = sum(1 for t in self._targets if t is None or (isinstance(t, float) and np.isnan(t)))
            if invalid_labels:
                results["warnings"].append(f"{invalid_labels} missing/invalid labels")
            results["checks"]["label_validity"] = invalid_labels == 0

        # Check missing values
        missing = sum(1 for s in self._sequences if not s or not s.strip())
        if missing:
            results["errors"].append(f"{missing} empty/missing sequences")
            results["valid"] = False
        results["checks"]["missing_values"] = missing == 0

        # Duplicate sequences
        seen = set()
        dup_count = 0
        for s in self._sequences:
            if s in seen:
                dup_count += 1
            else:
                seen.add(s)
        results["checks"]["no_duplicates"] = dup_count == 0
        if dup_count:
            results["warnings"].append(f"{dup_count} duplicate sequences in split '{self.split.value}'")

        results["valid"] = all(
            v is True for v in results["checks"].values()
        ) and len(results["errors"]) == 0

        if results["valid"]:
            logger.info(f"{self.DATASET_NAME}: integrity verification passed")
        else:
            for err in results["errors"]:
                logger.error(f"Verification error: {err}")
            for warn in results["warnings"]:
                logger.warning(f"Verification warning: {warn}")

        return results

    @abstractmethod
    def preprocess(self) -> Path:
        """Preprocess raw data into processed format (parquet/tensors).

        Must be implemented by subclasses.

        Returns:
            Path to processed data directory.
        """
        pass

    def save_processed_data(
        self,
        split_data: Dict[str, Dict[str, Any]],
    ) -> Path:
        """Save preprocessed split data to disk.

        Args:
            split_data: Dict split_name -> {"sequences": [...], "labels": [...]}.

        Returns:
            Path to processed directory.
        """
        import pandas as pd
        ensure_dir(self.processed_dir)

        for split_name, data in split_data.items():
            df = pd.DataFrame(data)
            path = self.processed_dir / f"{split_name}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} samples to {path}")

        return self.processed_dir

    def load_processed_data(self) -> Dict[str, Any]:
        """Load processed data from parquet files.

        Returns:
            Dict split_name -> {"sequences": [...], "labels": [...], ...}.
        """
        import pandas as pd

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"No processed data found at {self.processed_dir}. "
                f"Run preprocess() first."
            )

        result = {}
        for parquet_file in sorted(self.processed_dir.glob("*.parquet")):
            split_name = parquet_file.stem
            df = pd.read_parquet(parquet_file)
            result[split_name] = df.to_dict(orient="list")

        return result

    def cache(self) -> bool:
        """Check if processed cache exists and is valid.

        Returns:
            True if valid cache exists, False if preprocessing needed.
        """
        manifest = self.metadata_dir / "manifest.json"
        if not manifest.exists():
            return False

        processed_files = list(self.processed_dir.glob("*.parquet"))
        if not processed_files:
            return False

        logger.info(f"{self.DATASET_NAME}: found cached data ({len(processed_files)} files)")
        return True

    # ------------------------------------------------------------------
    # Statistics & Visualization
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Compute and save dataset statistics.

        Generates dataset_summary.csv in statistics_dir.

        Returns:
            Statistics dictionary.
        """
        ensure_dir(self.statistics_dir)

        # Collect all splits
        split_sequences: Dict[str, List[str]] = {}
        split_labels: Dict[str, List[Any]] = {}
        all_sequences: List[str] = []
        all_labels: List[Any] = []

        try:
            data = self.load_processed_data()
            for split_name, split_data in data.items():
                seqs = split_data.get("sequences", [])
                lbls = split_data.get("labels", split_data.get("targets", []))
                split_sequences[split_name] = seqs
                split_labels[split_name] = lbls
                all_sequences.extend(seqs)
                all_labels.extend(lbls)
        except FileNotFoundError:
            # Fall back to current instance data
            all_sequences = self._sequences
            all_labels = self._targets
            split_sequences = {self.split.value: self._sequences}
            split_labels = {self.split.value: self._targets}

        from ..statistics.statistics import compute_dataset_statistics, save_statistics_csv

        stats = compute_dataset_statistics(
            sequences=all_sequences,
            labels=all_labels if all_labels else None,
            split_names={k: list(range(len(v))) for k, v in split_sequences.items()},
        )

        csv_path = self.statistics_dir / "dataset_summary.csv"
        save_statistics_csv(stats, csv_path)
        logger.info(f"Saved statistics to {csv_path}")

        return stats

    def visualize(self) -> Dict[str, Path]:
        """Generate and save dataset visualization figures.

        Saves: length histogram, boxplot, class distribution,
        train/val/test comparison as PNG (300 dpi) + PDF.

        Returns:
            Dict of figure_name -> Path.
        """
        ensure_dir(self.figures_dir)

        all_sequences: List[str] = []
        all_labels: List[Any] = []
        split_sequences: Dict[str, List[str]] = {}
        split_labels: Dict[str, List[Any]] = {}

        try:
            data = self.load_processed_data()
            for sn, sd in data.items():
                seqs = sd.get("sequences", [])
                lbls = sd.get("labels", sd.get("targets", []))
                split_sequences[sn] = seqs
                split_labels[sn] = lbls
                all_sequences.extend(seqs)
                all_labels.extend(lbls)
        except FileNotFoundError:
            all_sequences = self._sequences
            all_labels = self._targets
            split_sequences = {self.split.value: self._sequences}
            split_labels = {self.split.value: self._targets}

        from ..visualization.visualization import generate_all_figures

        figures = generate_all_figures(
            sequences=all_sequences,
            labels=all_labels if all_labels else None,
            split_sequences=split_sequences if len(split_sequences) > 1 else None,
            split_labels=split_labels if len(split_labels) > 1 else None,
            output_dir=self.figures_dir,
        )

        logger.info(f"Generated {len(figures)} figure(s) in {self.figures_dir}")
        return figures

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate a manifest.json for this dataset.

        Contains: dataset version, preprocessing version, checksum,
        generation timestamp, framework version, git commit.

        Returns:
            Manifest dictionary.
        """
        import json

        from ..datasets.downloader import compute_sha256

        manifest = {
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "preprocessing_version": self.PREPROCESSING_VERSION,
            "generated_at": datetime.now().isoformat(),
            "framework_version": get_framework_version(),
            "git_commit": get_git_commit(),
            "checksums": {},
            "split_info": {
                "split": self.split.value,
                "num_sequences": len(self._sequences),
            },
        }

        # Compute checksums for processed files
        for pf in sorted(self.processed_dir.glob("*.parquet")):
            manifest["checksums"][pf.name] = compute_sha256(pf)

        ensure_dir(self.metadata_dir)
        manifest_path = self.metadata_dir / "manifest.json"
        save_json(manifest, manifest_path)
        logger.info(f"Saved manifest to {manifest_path}")

        return manifest

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset from disk. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return DatasetInfo describing this dataset."""
        pass

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_splits(
        self,
        tokenizer: Optional[Any] = None,
        max_seq_len: Optional[int] = None,
    ) -> Dict[str, "BaseDataset"]:
        """Return train/validation/test splits as BaseDataset instances.

        Args:
            tokenizer: Optional tokenizer for all splits.
            max_seq_len: Max sequence length for all splits.

        Returns:
            Dict of split_name -> BaseDataset instance.
        """
        max_seq_len = max_seq_len or self.max_seq_len
        splits: Dict[str, "BaseDataset"] = {}

        try:
            data = self.load_processed_data()
            for split_name in sorted(data.keys()):
                if split_name == "all":
                    continue
                # Create a new instance of the same class
                instance = type(self)(
                    data_dir=self.data_dir,
                    split=DatasetSplit(split_name) if split_name in ("train", "valid", "test") else DatasetSplit.TRAIN,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer or self.tokenizer,
                    config=self.config,
                )
                splits[split_name] = instance
        except FileNotFoundError:
            logger.warning("No processed data found for get_splits()")
            return {}

        return splits

    def _is_downloaded(self) -> bool:
        """Check if raw data already exists."""
        if not self.raw_dir.exists():
            return False
        if self.REQUIRED_FILES:
            from ..datasets.downloader import list_missing_files
            missing = list_missing_files(self.raw_dir, self.REQUIRED_FILES)
            return len(missing) == 0
        return any(self.raw_dir.iterdir())

    # ------------------------------------------------------------------
    # PyTorch Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sequence = self._sequences[idx]
        item = {"sequence": sequence, "index": idx}

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                sequence,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            item.update({k: v.squeeze(0) if v.dim() > 1 else v for k, v in encoding.items()})

        if idx < len(self._targets):
            target = self._targets[idx]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            item["target"] = target

        if idx < len(self._metadata):
            item["metadata"] = self._metadata[idx]

        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collate function for DataLoader."""
        if not batch:
            return {}
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            values = [item[key] for item in batch]
            if key == "target":
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values)
                else:
                    collated[key] = torch.tensor(values)
            elif isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    def get_sequences(self) -> List[str]:
        return self._sequences.copy()

    def get_targets(self) -> List[Any]:
        return self._targets.copy()