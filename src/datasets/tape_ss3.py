"""TAPE Secondary Structure (SS3) dataset — 3-class secondary structure prediction.

Download from the official TAPE benchmark repository.
Each split contains (sequence, 3-class label) pairs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..interfaces.base_dataset import (
    BaseDataset,
    DatasetInfo,
    DatasetSplit,
    TaskType,
)
from ..utils.logging import get_logger


logger = get_logger(__name__)


# SS3 labels: 0=H (helix), 1=E (strand), 2=C (coil)
SS3_LABEL_NAMES = {0: "Helix (H)", 1: "Strand (E)", 2: "Coil (C)"}


class TapeSS3Dataset(BaseDataset):
    """TAPE Secondary Structure (SS3) dataset.

    Official benchmark for 3-class secondary structure prediction.
    Each split file is a TSV with columns: id, sequence, ss3_label.

    Download URL:
        https://github.com/tatyanaeriksen/tape_dumps/raw/master/secondary_structure.tar.gz
    """

    DATASET_NAME = "tape_ss3"
    DATASET_VERSION = "1.0.0"
    PREPROCESSING_VERSION = "1.0.0"
    DOWNLOAD_URL = (
        "https://github.com/tatyanaeriksen/tape_dumps/raw/"
        "master/secondary_structure.tar.gz"
    )
    REQUIRED_FILES = {
        "secondary_structure/train.tsv": None,
        "secondary_structure/valid.tsv": None,
        "secondary_structure/test.tsv": None,
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_seq_len: int = 1022,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._info: Optional[DatasetInfo] = None
        super().__init__(data_dir, split, max_seq_len, tokenizer, config)

    def _load_data(self) -> None:
        """Load the appropriate split from processed parquet or raw TSV."""
        processed_file = self.processed_dir / f"{self.split.value}.parquet"
        if processed_file.exists():
            df = pd.read_parquet(processed_file)
            self._sequences = df["sequence"].tolist()
            self._targets = df["ss3_label"].tolist()
            if "id" in df.columns:
                self._metadata = df[["id"]].to_dict(orient="records")
            logger.info(
                f"Loaded {len(self._sequences)} samples from {processed_file.name}"
            )
            return

        # Fall back to raw TSV
        raw_file = self.raw_dir / "secondary_structure" / f"{self.split.value}.tsv"
        if not raw_file.exists():
            logger.warning(
                f"No data found for {self.DATASET_NAME}/{self.split.value}. "
                f"Call download() and preprocess() first."
            )
            return

        df = pd.read_csv(raw_file, sep="\t", header=0)
        self._sequences = df.iloc[:, 1].astype(str).tolist()
        self._targets = df.iloc[:, 2].tolist()
        self._metadata = [{"id": str(row[0])} for row in df.itertuples(index=False)]

        logger.info(
            f"Loaded {len(self._sequences)} samples from raw {raw_file.name}"
        )

    def get_info(self) -> DatasetInfo:
        if self._info is None:
            self._info = DatasetInfo(
                name=self.DATASET_NAME,
                task_type=TaskType.TOKEN_CLASSIFICATION,
                num_classes=3,
                description=(
                    "TAPE Secondary Structure (SS3) — 3-class per-residue "
                    "secondary structure prediction (Helix/Strand/Coil)"
                ),
            )
        return self._info

    def preprocess(self) -> Path:
        """Parse raw TSV files and save as parquet.

        The TAPE SS3 dataset contains per-residue labels (one per amino acid).
        For benchmarking we convert to sequence-level format where each
        sequence has an associated ss3_class (majority vote).

        Returns:
            Path to processed directory.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        split_data: Dict[str, Dict[str, list]] = {}
        for split_name in ("train", "valid", "test"):
            raw_file = self.raw_dir / "secondary_structure" / f"{split_name}.tsv"
            if not raw_file.exists():
                logger.warning(f"Raw file not found: {raw_file}")
                continue

            df = pd.read_csv(raw_file, sep="\t", header=0)
            df.columns = ["id", "sequence", "ss3_label"]

            # Validate labels
            valid_mask = df["ss3_label"].isin([0, 1, 2])
            if not valid_mask.all():
                invalid = (~valid_mask).sum()
                logger.warning(f"{split_name}: {invalid} invalid labels, filtering out")
                df = df[valid_mask]

            split_data[split_name] = {
                "id": df["id"].tolist(),
                "sequence": df["sequence"].astype(str).tolist(),
                "ss3_label": df["ss3_label"].tolist(),
            }

            # Per-residue labels for token classification
            if "ss3_labels_per_residue" in self.config.get("store_residue_labels", False):
                split_data[split_name]["ss3_labels_per_residue"] = [
                    [int(c) for c in seq_label]
                    for seq_label in df["ss3_label"].tolist()
                ]

            logger.info(f"Preprocessed {split_name}: {len(df)} sequences")

        return self.save_processed_data(split_data)

    def verify(self) -> Dict[str, Any]:
        """Extend base verification with SS3-specific checks."""
        results = super().verify()

        if self._targets:
            import numpy as np
            labels = np.array(self._targets)
            unique = np.unique(labels)
            invalid = set(unique.tolist()) - {0, 1, 2}
            if invalid:
                results["errors"].append(f"Invalid SS3 labels found: {invalid}")
                results["valid"] = False
            results["checks"]["ss3_labels_valid"] = len(invalid) == 0

            # Check label balance
            counts = {k: int((labels == k).sum()) for k in (0, 1, 2)}
            total = len(labels)
            for k, c in counts.items():
                pct = 100 * c / total
                if pct < 10:
                    results["warnings"].append(
                        f"SS3 class '{SS3_LABEL_NAMES[k]}' has only {pct:.1f}% "
                        f"of samples in split '{self.split.value}'"
                    )

        return results

    def statistics(self) -> Dict[str, Any]:
        """Compute SS3-specific statistics."""
        stats = super().statistics()

        try:
            data = self.load_processed_data()
        except FileNotFoundError:
            data = {}

        # Per-residue label distribution if available
        all_residue_labels = []
        for split_name, split_data in data.items():
            if "ss3_labels_per_residue" in split_data:
                for labels in split_data["ss3_labels_per_residue"]:
                    all_residue_labels.extend(labels)

        if all_residue_labels:
            import numpy as np
            unique, counts = np.unique(all_residue_labels, return_counts=True)
            res_dist = {}
            for u, c in zip(unique, counts):
                name = SS3_LABEL_NAMES.get(int(u), str(u))
                res_dist[f"residue_{name}"] = int(c)
            stats["residue_label_distribution"] = res_dist

        return stats

    def get_splits(
        self,
        tokenizer: Optional[Any] = None,
        max_seq_len: Optional[int] = None,
    ) -> Dict[str, "TapeSS3Dataset"]:
        """Return train/valid/test splits."""
        max_seq_len = max_seq_len or self.max_seq_len

        splits = {
            "train": TapeSS3Dataset(
                data_dir=self.data_dir,
                split=DatasetSplit.TRAIN,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer or self.tokenizer,
                config=self.config,
            ),
            "valid": TapeSS3Dataset(
                data_dir=self.data_dir,
                split=DatasetSplit.VALIDATION,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer or self.tokenizer,
                config=self.config,
            ),
            "test": TapeSS3Dataset(
                data_dir=self.data_dir,
                split=DatasetSplit.TEST,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer or self.tokenizer,
                config=self.config,
            ),
        }
        return splits
