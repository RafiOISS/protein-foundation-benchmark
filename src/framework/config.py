"""ExperimentConfig — dataclass-based configuration for benchmark experiments.

Replaces raw dictionaries and Hydra with validated dataclass config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentConfig:
    """Complete configuration for a benchmark experiment.

    All fields are validated via type hints. No raw dictionaries for
    top-level configuration.
    """

    # Experiment identity
    name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Dataset
    dataset: str = ""
    task: str = ""  # regression, binary_classification, multiclass_classification, etc.
    max_seq_len: int = 1022
    dataset_dir: Optional[Union[str, Path]] = None

    # Models
    models: List[str] = field(default_factory=list)
    device: str = "auto"
    seed: int = 42

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5
    freeze_strategy: str = "none"

    # Paths
    project_root: Optional[Union[str, Path]] = None
    output_dir: Optional[Union[str, Path]] = None
    cache_dir: Optional[Union[str, Path]] = None

    # Metrics
    metrics: List[str] = field(default_factory=list)

    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)

    VALID_TASKS = {"regression", "binary_classification", "multiclass_classification"}
    VALID_DEVICES = {"auto", "cpu", "cuda"}

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("ExperimentConfig.dataset must be set")
        if not self.task:
            raise ValueError("ExperimentConfig.task must be set")
        if self.task not in self.VALID_TASKS:
            raise ValueError(f"ExperimentConfig.task must be one of {self.VALID_TASKS}, got '{self.task}'")
        if not self.models:
            raise ValueError("ExperimentConfig.models must contain at least one model")

        device_lower = self.device.lower().split(":")[0]
        if device_lower not in self.VALID_DEVICES:
            raise ValueError(f"ExperimentConfig.device must be one of {self.VALID_DEVICES}, got '{self.device}'")

        if self.project_root:
            self.project_root = Path(self.project_root)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
        if self.dataset_dir:
            self.dataset_dir = Path(self.dataset_dir)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        from ..utils.io import load_yaml
        data = load_yaml(path)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                d[key] = str(value)
            else:
                d[key] = value
        return d