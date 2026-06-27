"""Dataset registry for the Protein Foundation Model Benchmark Framework.

Allows datasets to be registered by name and instantiated dynamically.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces.base_dataset import BaseDataset, DatasetSplit
from ..utils.logging import get_logger


logger = get_logger(__name__)


class DatasetRegistry:
    """Registry for benchmark datasets.

    Datasets register themselves with a name and are instantiated on demand.
    """

    _datasets: Dict[str, Type[BaseDataset]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        dataset_class: Type[BaseDataset],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a dataset class.

        Args:
            name: Unique dataset identifier (e.g., 'tape_ss3', 'fluorescence').
            dataset_class: Dataset class (must subclass BaseDataset).
            config: Optional default configuration.
        """
        if not issubclass(dataset_class, BaseDataset):
            raise TypeError(f"{dataset_class.__name__} must subclass BaseDataset")

        cls._datasets[name] = dataset_class
        if config:
            cls._configs[name] = config

        logger.info(f"Registered dataset '{name}' -> {dataset_class.__name__}")

    @classmethod
    def create(
        cls,
        name: str,
        data_dir: Union[str, Path],
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_seq_len: int = 1022,
        tokenizer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseDataset:
        """Create a dataset instance by name.

        Args:
            name: Registered dataset name.
            data_dir: Path to dataset files.
            split: Dataset split.
            max_seq_len: Maximum sequence length.
            tokenizer: Optional tokenizer.
            config: Dataset configuration (merged with defaults).
            **kwargs: Additional arguments.

        Returns:
            Dataset instance.
        """
        if name not in cls._datasets:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {list(cls._datasets.keys())}"
            )

        dataset_class = cls._datasets[name]
        merged_config = {**cls._configs.get(name, {}), **(config or {})}

        logger.info(f"Creating dataset '{name}' ({split.value})")
        return dataset_class(
            data_dir=Path(data_dir) / name,
            split=split,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            config=merged_config,
            **kwargs,
        )

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._datasets.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[BaseDataset]:
        """Get the dataset class for a registered name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset '{name}'")
        return cls._datasets[name]

    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """Get default config for a registered dataset."""
        return cls._configs.get(name, {}).copy()

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a dataset."""
        cls._datasets.pop(name, None)
        cls._configs.pop(name, None)
        logger.info(f"Unregistered dataset '{name}'")