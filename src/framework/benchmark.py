"""ProteinBenchmark — the single public API for the benchmark framework.

Usage:
    benchmark = ProteinBenchmark()
    benchmark.register_model("esm2", ESM2)
    benchmark.register_dataset("tape_ss3", TAPESS3Dataset)

    config = ExperimentConfig(name="my_exp", dataset="tape_ss3", models=["esm2"])
    results = benchmark.run(config)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import setup_logging, get_logger
from ..utils.seed import set_seed
from ..framework.config import ExperimentConfig
from ..framework.experiment import Experiment
from ..framework.pipeline import Pipeline
from ..registry.model_registry import ModelRegistry
from ..registry.dataset_registry import DatasetRegistry


logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single model + dataset benchmark run."""
    experiment_id: str
    experiment_name: str
    model_name: str
    dataset_name: str
    task: str
    metrics: Dict[str, float]
    duration_seconds: float
    status: str
    config: Dict[str, Any] = field(default_factory=dict)


class ProteinBenchmark:
    """Main public interface for the benchmark framework.

    Notebooks call benchmark.run(config) — nothing else.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        seed: int = 42,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the benchmark framework.

        Args:
            project_root: Project root (auto-detected if None).
            seed: Random seed for reproducibility.
            log_level: Logging level.
        """
        setup_logging(level=log_level)

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
            if project_root is None
            else Path(project_root)
        )
        self.seed = seed
        set_seed(seed)

        self._experiments: Dict[str, Experiment] = {}

        logger.info(f"ProteinBenchmark initialized (root={self.project_root})")

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment from a config."""
        exp = Experiment(
            name=config.name,
            config=config.to_dict(),
            project_root=self.project_root,
            seed=config.seed,
        )
        self._experiments[exp.id] = exp
        logger.info(f"Created experiment: {config.name} (id={exp.id})")
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        return list(self._experiments.values())

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register_model(self, name: str, model_class: type, config: Optional[Dict] = None) -> None:
        ModelRegistry.register(name, model_class, config)

    def register_dataset(self, name: str, dataset_class: type, config: Optional[Dict] = None) -> None:
        DatasetRegistry.register(name, dataset_class, config)

    def list_models(self) -> List[str]:
        return ModelRegistry.list_models()

    def list_datasets(self) -> List[str]:
        return DatasetRegistry.list_datasets()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, config: ExperimentConfig) -> List[BenchmarkResult]:
        """Run a benchmark experiment.

        Args:
            config: Fully specified ExperimentConfig.

        Returns:
            List of BenchmarkResult, one per model in config.models.
        """
        exp = self.create_experiment(config)
        exp.start()
        results = []

        data_dir = config.dataset_dir or (self.project_root / "outputs" / "cache" / "datasets")
        device = config.device

        for model_name in config.models:
            logger.info(f"Running model={model_name}  dataset={config.dataset}")

            try:
                model = ModelRegistry.create(model_name, device=device)
                tokenizer = getattr(model, "get_tokenizer", lambda: None)()

                train_ds = DatasetRegistry.create(
                    config.dataset, data_dir=data_dir, split="train",
                    max_seq_len=config.max_seq_len, tokenizer=tokenizer,
                )
                val_ds = DatasetRegistry.create(
                    config.dataset, data_dir=data_dir, split="valid",
                    max_seq_len=config.max_seq_len, tokenizer=tokenizer,
                )
                test_ds = DatasetRegistry.create(
                    config.dataset, data_dir=data_dir, split="test",
                    max_seq_len=config.max_seq_len, tokenizer=tokenizer,
                )

                pipeline = Pipeline(
                    model=model,
                    experiment=exp,
                    dataset_name=config.dataset,
                    model_name=model_name,
                    device=device,
                )

                pipeline_result = pipeline.run(
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    test_dataset=test_ds,
                )

                result = BenchmarkResult(
                    experiment_id=exp.id,
                    experiment_name=config.name,
                    model_name=model_name,
                    dataset_name=config.dataset,
                    task=config.task,
                    metrics=pipeline_result.get("metrics", {}),
                    duration_seconds=pipeline_result.get("duration", 0.0),
                    status="completed",
                    config=config.to_dict(),
                )

            except Exception as e:
                logger.error(f"Benchmark failed for {model_name} on {config.dataset}: {e}")
                result = BenchmarkResult(
                    experiment_id=exp.id,
                    experiment_name=config.name,
                    model_name=model_name,
                    dataset_name=config.dataset,
                    task=config.task,
                    metrics={},
                    duration_seconds=0.0,
                    status=f"failed: {e}",
                    config=config.to_dict(),
                )

            results.append(result)

        exp.complete()
        return results

    def info(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "seed": self.seed,
            "registered_models": self.list_models(),
            "registered_datasets": self.list_datasets(),
            "experiments": len(self._experiments),
        }