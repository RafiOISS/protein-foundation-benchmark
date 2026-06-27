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
    def load(
        cls,
        name: str,
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        download_if_missing: bool = True,
        preprocess_if_needed: bool = True,
        verify_on_load: bool = True,
        **kwargs,
    ) -> BaseDataset:
        """Load a dataset with full lifecycle: download → verify → preprocess → cache.

        This is the primary entry point for loading datasets in the framework.

        Args:
            name: Registered dataset name.
            data_dir: Directory containing the dataset (default: cache_dir/name).
            cache_dir: Root cache directory (used with name to form data_dir).
            download_if_missing: Download raw data if not present.
            preprocess_if_needed: Preprocess if processed cache doesn't exist.
            verify_on_load: Run integrity verification after loading.
            **kwargs: Additional arguments passed to create().

        Returns:
            Dataset instance (train split).
        """
        if data_dir is None:
            if cache_dir is None:
                cache_dir = Path("outputs/cache/datasets")
            data_dir = Path(cache_dir) / name

        data_dir = Path(data_dir)

        if name not in cls._datasets:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {list(cls._datasets.keys())}"
            )

        dataset_class = cls._datasets[name]
        merged_config = {**cls._configs.get(name, {}), **(kwargs.pop("config", {}))}

        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        metadata_dir = data_dir / "metadata"

        has_raw = (
            all((raw_dir / rel).exists() for rel in dataset_class.REQUIRED_FILES)
            if dataset_class.REQUIRED_FILES
            else raw_dir.exists()
        )

        # Step 1: Download if missing
        if download_if_missing and not has_raw:
            logger.info(f"Downloading dataset '{name}' to {raw_dir}")
            tmp = dataset_class(
                data_dir=data_dir, split=DatasetSplit.TRAIN,
                config=merged_config, **kwargs,
            )
            tmp.download()
            has_raw = True
        elif not has_raw:
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {raw_dir}. "
                f"Set download_if_missing=True or download manually."
            )

        # Step 2: Preprocess if no processed cache
        if preprocess_if_needed and not list(processed_dir.glob("*.parquet")):
            tmp = dataset_class(
                data_dir=data_dir, split=DatasetSplit.TRAIN,
                config=merged_config, **kwargs,
            )
            tmp.preprocess()
            if verify_on_load:
                result = tmp.verify()
                if not result.get("valid", True):
                    for err in result.get("errors", []):
                        logger.error(f"Verification error: {err}")
            tmp.generate_manifest()

        # Step 3: Create final instance with data loaded
        ds = dataset_class(
            data_dir=data_dir, split=DatasetSplit.TRAIN,
            config=merged_config, **kwargs,
        )

        logger.info(f"Dataset '{name}' loaded successfully ({len(ds)} samples)")
        return ds

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