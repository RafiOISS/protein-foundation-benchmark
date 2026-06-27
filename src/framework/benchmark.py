"""ProteinBenchmark — the main public API for the benchmark framework.

Users interact with this single class to run benchmarks, manage experiments,
and access results. Everything else is internal.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..utils.logging import setup_logging, get_logger
from ..utils.io import load_yaml
from ..utils.seed import set_seed
from ..utils.environment import Environment
from ..framework.experiment import Experiment, ExperimentStatus
from ..framework.pipeline import Pipeline, PipelineStage
from ..registry.model_registry import ModelRegistry
from ..registry.dataset_registry import DatasetRegistry


logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    experiment_id: str
    experiment_name: str
    model_name: str
    dataset_name: str
    task_type: str
    metrics: Dict[str, float]
    duration_seconds: float
    status: str
    config: Dict[str, Any] = field(default_factory=dict)


class ProteinBenchmark:
    """Main public interface for the benchmark framework.

    Usage:
        benchmark = ProteinBenchmark()
        benchmark.run(
            experiment="my_exp",
            dataset="fluorescence",
            models=["esm2", "protbert"],
        )
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the benchmark framework.

        Args:
            project_root: Project root directory (auto-detected if None).
            config_dir: Configuration directory (defaults to project_root/configs).
            seed: Random seed for reproducibility.
            log_level: Logging level.
        """
        setup_logging(level=log_level)

        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.config_dir = Path(config_dir) if config_dir else self.project_root / "configs"
        self.seed = seed

        set_seed(seed)

        self._experiments: Dict[str, Experiment] = {}
        self._pipelines: Dict[str, Pipeline] = {}

        logger.info(f"ProteinBenchmark initialized (root={self.project_root})")

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def load_config(self, name: str) -> Dict[str, Any]:
        """Load a YAML config from configs/framework/, configs/datasets/, etc.

        Args:
            name: Config name (e.g., 'framework/default', 'datasets/fluorescence').

        Returns:
            Parsed configuration dictionary.
        """
        path = self.config_dir / f"{name}.yaml"
        if not path.exists():
            logger.warning(f"Config not found: {path}")
            return {}
        return load_yaml(path)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            description: Description.
            tags: Tags for categorization.
            config: Experiment configuration.

        Returns:
            Experiment instance.
        """
        experiment = Experiment(
            name=name,
            description=description,
            tags=tags or [],
            config=config or {},
            project_root=self.project_root,
            seed=self.seed,
        )
        self._experiments[experiment.id] = experiment
        logger.info(f"Created experiment: {name} (id={experiment.id})")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        return list(self._experiments.values())

    # ------------------------------------------------------------------
    # Registry passthrough
    # ------------------------------------------------------------------

    def register_model(self, name: str, model_class: type, config: Optional[Dict] = None) -> None:
        """Register a model class."""
        ModelRegistry.register(name, model_class, config)

    def register_dataset(self, name: str, dataset_class: type, config: Optional[Dict] = None) -> None:
        """Register a dataset class."""
        DatasetRegistry.register(name, dataset_class, config)

    def list_models(self) -> List[str]:
        """List registered models."""
        return ModelRegistry.list_models()

    def list_datasets(self) -> List[str]:
        """List registered datasets."""
        return DatasetRegistry.list_datasets()

    # ------------------------------------------------------------------
    # Main benchmark execution
    # ------------------------------------------------------------------

    def run(
        self,
        experiment: Union[str, Experiment],
        dataset: str,
        models: List[str],
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ) -> List[BenchmarkResult]:
        """Run a benchmark experiment.

        Args:
            experiment: Experiment name (string) or Experiment instance.
            dataset: Dataset name (must be registered).
            models: List of model names (must be registered).
            config: Additional configuration overrides.
            device: Device to use ('auto', 'cuda', 'cpu').

        Returns:
            List of BenchmarkResult objects.
        """
        # Resolve experiment
        if isinstance(experiment, str):
            exp = self.create_experiment(name=experiment, config=config)
        else:
            exp = experiment

        exp.start()
        results = []

        for model_name in models:
            logger.info(f"Running: model={model_name}, dataset={dataset}")

            try:
                # Create model
                model = ModelRegistry.create(model_name, device=device)

                # Create dataset loaders
                train_dataset = DatasetRegistry.create(
                    dataset,
                    data_dir=self.project_root / "outputs" / "cache" / "datasets",
                    split="train",
                    tokenizer=getattr(model, "get_tokenizer", lambda: None)(),
                )
                val_dataset = DatasetRegistry.create(
                    dataset,
                    data_dir=self.project_root / "outputs" / "cache" / "datasets",
                    split="valid",
                    tokenizer=getattr(model, "get_tokenizer", lambda: None)(),
                )
                test_dataset = DatasetRegistry.create(
                    dataset,
                    data_dir=self.project_root / "outputs" / "cache" / "datasets",
                    split="test",
                    tokenizer=getattr(model, "get_tokenizer", lambda: None)(),
                )

                # Run pipeline
                pipeline = Pipeline(
                    model=model,
                    experiment=exp,
                    dataset_name=dataset,
                    model_name=model_name,
                    device=device,
                )

                pipeline_result = pipeline.run(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                )

                result = BenchmarkResult(
                    experiment_id=exp.id,
                    experiment_name=exp.name,
                    model_name=model_name,
                    dataset_name=dataset,
                    task_type=train_dataset.get_info().task_type.value,
                    metrics=pipeline_result.get("metrics", {}),
                    duration_seconds=pipeline_result.get("duration", 0.0),
                    status="completed",
                )

            except Exception as e:
                logger.error(f"Benchmark failed for {model_name} on {dataset}: {e}")
                result = BenchmarkResult(
                    experiment_id=exp.id,
                    experiment_name=exp.name,
                    model_name=model_name,
                    dataset_name=dataset,
                    task_type="unknown",
                    metrics={},
                    duration_seconds=0.0,
                    status=f"failed: {e}",
                )

            results.append(result)

        exp.complete()
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return framework summary information."""
        env = Environment()
        return {
            "project_root": str(self.project_root),
            "seed": self.seed,
            "registered_models": self.list_models(),
            "registered_datasets": self.list_datasets(),
            "experiments": len(self._experiments),
            "environment": env.capture(),
        }