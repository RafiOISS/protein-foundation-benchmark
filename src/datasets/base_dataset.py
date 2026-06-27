"""Concrete dataset utilities and base class re-exports."""

from ..interfaces.base_dataset import BaseDataset, DatasetSplit, TaskType, DatasetInfo
from ..utils.logging import get_logger

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = get_logger(__name__)


class TabularDataset(BaseDataset):
    """Dataset loaded from CSV/TSV/Parquet files."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_seq_len: int = 1022,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        file_pattern: str = "{split}.csv",
        sequence_column: str = "sequence",
        target_column: str = "target",
    ) -> None:
        self.file_pattern = file_pattern
        self.sequence_column = sequence_column
        self.target_column = target_column
        super().__init__(data_dir, split, max_seq_len, tokenizer, config)

    def _load_data(self) -> None:
        import pandas as pd
        file_path = self.data_dir / self.file_pattern.format(split=self.split.value)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        df = pd.read_csv(file_path)
        if self.sequence_column not in df.columns:
            raise ValueError(f"Column '{self.sequence_column}' not in {file_path}")
        self._sequences = df[self.sequence_column].astype(str).tolist()
        self._targets = df[self.target_column].tolist() if self.target_column in df.columns else [None] * len(self._sequences)

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(name=self.data_dir.name, task_type=TaskType.CUSTOM)


class InMemoryDataset(BaseDataset):
    """Dataset created from in-memory data."""

    def __init__(
        self,
        sequences: List[str],
        targets: Optional[List[Any]] = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_seq_len: int = 1022,
        tokenizer: Optional[Any] = None,
        task_type: TaskType = TaskType.CUSTOM,
    ) -> None:
        self._sequences_input = sequences
        self._targets_input = targets
        self._task_type = task_type
        super().__init__(None, split, max_seq_len, tokenizer)

    def _load_data(self) -> None:
        self._sequences = self._sequences_input
        self._targets = self._targets_input or [None] * len(self._sequences)

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(name="in_memory", task_type=self._task_type)



__all__ = [
    "BaseDataset",
    "DatasetSplit",
    "TaskType",
    "DatasetInfo",
    "TabularDataset",
    "InMemoryDataset",
]