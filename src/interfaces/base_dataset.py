"""Base dataset interface for the Protein Foundation Model Benchmark Framework.

All benchmark datasets must inherit from BaseDataset.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch
from torch.utils.data import Dataset

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


class BaseDataset(Dataset, ABC):
    """Abstract base class for protein sequence datasets.

    All benchmark datasets must inherit from this class and implement
    the abstract methods.
    """

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

        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset from disk. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return DatasetInfo describing this dataset."""
        pass

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