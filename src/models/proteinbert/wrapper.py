"""ProteinBERTModel — wrapper for ProteinBERT with lazy TensorFlow imports.

Responsibilities:
  - lazy import TensorFlow and proteinbert packages
  - instantiate and manage the pretrained model
  - expose placeholder interfaces: load(), predict(), save(), summary()
  - preprocessing: prepare_inputs() — convert raw dataset to model-ready batches
  - no training logic
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel
from .loader import load_pretrained, locate_checkpoints, download_if_missing


logger = get_logger(__name__)


class ProteinBERTModel(BaseProteinModel):
    """Wrapper for ProteinBERT with TensorFlow backend.

    ProteinBERT requires TensorFlow and the proteinbert package.
    These are imported lazily — only inside methods that actually
    need them. Framework-level imports never trigger TF loading.

    Placeholder methods:
      load()   — load model from disk
      predict() — run inference (to be implemented in Phase 2.2+)
      save()   — persist model to disk
      summary() — print model architecture summary
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize ProteinBERTModel.

        Args:
            model_path: Path to pretrained ProteinBERT weights.
            config: Model configuration.
            device: Device to run on ('auto', 'cpu', 'cuda').
            **kwargs: Additional keyword arguments.
        """
        config = config or {}
        config.setdefault("model_type", "proteinbert")
        config.setdefault("max_seq_len", 512)
        config.setdefault("embedding_dim", 768)
        config.setdefault("batch_size", 8)
        config.setdefault("learning_rate", 1e-4)
        config.setdefault("epochs", 20)
        config.setdefault("mixed_precision", False)
        config.setdefault("seed", 42)

        super().__init__(config, device)
        self.model_path = Path(model_path) if model_path else None
        self._tf_model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False
        self._cache_manager: Any = None
        self._runtime: Any = None

        logger.info("ProteinBERTModel initialized (TensorFlow backend not yet loaded)")

    # ------------------------------------------------------------------
    # Preprocessing + adapter
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        dataset: Any,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Preprocess a TAPE SS3 dataset into ProteinBERT-ready inputs.

        Runs the full preprocessing pipeline: validation, encoding,
        padding, truncation, batching, statistics, figures, and report.

        Args:
            dataset: TapeSS3Dataset instance with loaded data.
            output_dir: Output directory for preprocessing artifacts.
                       If None, uses outputs/preprocessing/.

        Returns:
            Dict with preprocessing results:
              - stats: all statistics
              - figures: dict of figure_name -> path
              - report: path to preprocessing report
              - encoded_shape: shape of encoded input array
              - num_valid_sequences: number of valid sequences
        """
        from .preprocessing import PreprocessingPipeline

        # Build preprocessing config from model config
        preproc_config = {
            "padding": self.config.get("padding", "right"),
            "truncation": self.config.get("truncation", True),
            "mask_padding": self.config.get("mask_padding", True),
            "max_length": self.config.get("max_seq_len", 512),
            "unknown_token": self.config.get("unknown_token", "X"),
            "extended_alphabet": self.config.get("extended_alphabet", False),
            "batch_size": self.config.get("batch_size", 8),
            "shuffle": False,
            "seed": self.config.get("seed", 42),
        }

        pipeline = PreprocessingPipeline(
            config=preproc_config,
            output_dir=output_dir,
        )

        if dataset is not None:
            result = pipeline.run(dataset)
        else:
            # Synthetic data for testing / dry-run
            result = pipeline.run_on_sequences(
                sequences=["ACDEFGHIKL" * 10, "MNPQRSTVWY" * 5],
                labels=["H", "E"],
            )
        logger.info(f"prepare_inputs complete: {result['num_valid_sequences']} sequences prepared")

        return result

    def prepare_dataset(
        self,
        sequences: List[str],
        labels: Optional[List[Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_id: Optional[str] = None,
    ) -> "ProteinBERTDataAdapter":
        """Preprocess raw sequences and create a validated data adapter.

        One-stop method: preprocessing → encoding → padding → adapter.
        Returns a ProteinBERTDataAdapter ready for iteration or TF dataset creation.

        Args:
            sequences: Raw protein sequence strings.
            labels: Optional SS3 label strings.
            output_dir: Explicit output directory for all adapter artifacts.
                        If None, uses outputs/experiments/<experiment_id>/adapter/.
            experiment_id: Experiment ID for structured output under
                           outputs/experiments/<experiment_id>/adapter/.
                           Only used if output_dir is None.

        Returns:
            Configured ProteinBERTDataAdapter instance.
        """
        from .adapter import ProteinBERTDataAdapter, _build_experiment_path

        adapter_config = {
            "batch_size": self.config.get("batch_size", 8),
            "max_length": self.config.get("max_seq_len", 512),
            "padding": self.config.get("padding", "right"),
            "truncation": self.config.get("truncation", True),
            "extended_alphabet": self.config.get("extended_alphabet", False),
            "unknown_token": self.config.get("unknown_token", "X"),
            "shuffle": False,
            "seed": self.config.get("seed", 42),
        }

        adapter = ProteinBERTDataAdapter(config=adapter_config)
        adapter.create_batches(sequences, labels)

        # Save all artifacts
        adapter.save_all(output_dir=output_dir, experiment_id=experiment_id)

        logger.info(
            f"prepare_dataset complete: {len(adapter)} batches, "
            f"{adapter.info()['num_samples']} samples"
        )

        return adapter

    # ------------------------------------------------------------------
    # Runtime initialization
    # ------------------------------------------------------------------

    def initialize_runtime(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> "Runtime":
        """Initialize runtime with full validation.

        Initialization order:
          1. Load configuration
          2. Initialize CacheManager
          3. Create workspace directories
          4. Verify directory permissions
          5. Verify available disk space
          6. Configure cache environment variables
          7. Validate environment
          8. Set deterministic seeds
          9. Configure GPU

        No TensorFlow imports during initialization.

        Args:
            workspace_root: Project workspace root (auto-detected if None).
            runtime_config: Runtime configuration overrides.

        Returns:
            Runtime instance.
        """
        from .runtime import Runtime

        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent.parent

        runtime = Runtime(
            workspace_root=workspace_root,
            config=runtime_config,
        )
        runtime.initialize()
        self._runtime = runtime
        self._cache_manager = runtime.cache_manager

        logger.info(f"Runtime initialized (workspace={workspace_root})")
        return runtime

    @property
    def runtime(self) -> Any:
        """Get the Runtime instance (None if not initialized)."""
        return self._runtime

    @property
    def cache_manager(self) -> Any:
        """Get the CacheManager instance (None if not initialized)."""
        return self._cache_manager

    def validate_environment(self) -> Dict[str, Any]:
        """Run pre-flight environment validation.

        Returns:
            Validation results dict.

        Raises:
            RuntimeError: If validation fails.
        """
        from ...utils.environment import validate_environment

        ws = Path(__file__).resolve().parent.parent.parent.parent
        cache_root = ws / "outputs" / "cache"
        return validate_environment(
            workspace_root=ws,
            cache_root=cache_root,
            min_disk_gb=self.config.get("min_disk_space_gb", 5.0),
            require_gpu=False,
        )

    def generate_runtime_reports(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Generate runtime evidence reports.

        Args:
            output_dir: Output directory (default: outputs/runtime/).

        Returns:
            Dict of artifact name -> Path.

        Raises:
            RuntimeError: If runtime not initialized.
        """
        if self._runtime is None:
            raise RuntimeError(
                "Runtime not initialized. Call initialize_runtime() first."
            )
        return self._runtime.generate_reports(output_dir=output_dir)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load TensorFlow/ProteinBERT if not already loaded."""
        if self._loaded:
            return
        self._tf_model, self._tokenizer = load_pretrained(
            model_path=self.model_path,
            config=self.config,
        )
        self._loaded = True

    # ------------------------------------------------------------------
    # Placeholder lifecycle methods
    # ------------------------------------------------------------------

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load model from disk.

        Args:
            path: Path to model checkpoint. Uses self.model_path if None.
        """
        path = Path(path) if path else self.model_path
        if path is None:
            logger.warning("No path provided to load(); using default pretrained model")
            self._ensure_loaded()
            return

        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        self.model_path = path
        self._loaded = False
        self._ensure_loaded()
        logger.info(f"Model loaded from {path}")

    def predict(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Run inference on input sequences.

        Placeholder — full implementation deferred to Phase 2.2+.

        Args:
            sequences: List of protein sequences.

        Returns:
            List of prediction dicts.
        """
        self._ensure_loaded()
        logger.warning("predict() is a placeholder — returns dummy output")
        return [{"sequence": s, "prediction": None} for s in sequences]

    def save(self, path: Union[str, Path]) -> Path:
        """Save model checkpoint to disk.

        Args:
            path: Output directory path.

        Returns:
            Path to saved checkpoint directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._tf_model is not None:
            import tensorflow as tf
            checkpoint_path = str(path / "proteinbert_weights.h5")
            self._tf_model.save_weights(checkpoint_path)
            logger.info(f"Saved ProteinBERT weights to {checkpoint_path}")
        else:
            logger.warning("No loaded model to save; call load() first")

        return path

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the model architecture and configuration.

        Returns:
            Dictionary with model metadata.
        """
        info: Dict[str, Any] = {
            "model_type": "proteinbert",
            "framework": "tensorflow",
            "loaded": self._loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "embedding_dim": self.get_embedding_dim(),
            "max_seq_len": self.get_max_seq_len(),
            "num_layers": self.get_num_layers(),
            "config": dict(self.config),
        }

        if self._loaded and self._tf_model is not None:
            try:
                total_params = sum(
                    w.size for w in self._tf_model.weights
                )
                trainable_params = sum(
                    w.size for w in self._tf_model.trainable_weights
                )
                info["total_parameters"] = total_params
                info["trainable_parameters"] = trainable_params
            except Exception:
                pass

        return info

    # ------------------------------------------------------------------
    # BaseProteinModel abstract methods
    # ------------------------------------------------------------------

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through ProteinBERT.

        Args:
            sequences: List of protein sequences.

        Returns:
            Model outputs as PyTorch tensor.
        """
        self._ensure_loaded()

        encoded = self._tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            max_length=self.config.get("max_seq_len", 512),
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        import tensorflow as tf
        outputs = self._tf_model(encoded["input_ids"])
        return torch.from_numpy(outputs.numpy())

    def extract_embeddings(
        self,
        sequences: List[str],
        layers: Union[str, List[int]] = "last",
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract per-sequence embeddings.

        Args:
            sequences: List of protein sequences.
            layers: Ignored for ProteinBERT (only last layer available).
            pooling: Pooling strategy ('mean', 'cls', 'max').

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        embeddings = self.forward(sequences)

        if pooling == "cls":
            embeddings = embeddings[:, 0, :]
        elif pooling == "mean":
            embeddings = embeddings.mean(dim=1)
        elif pooling == "max":
            embeddings = embeddings.max(dim=1).values

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 768)

    def get_max_seq_len(self) -> int:
        return self.config.get("max_seq_len", 512)

    def get_num_layers(self) -> int:
        return 12

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "ProteinBERTModel":
        """Load pretrained ProteinBERT.

        Args:
            model_path: Optional path to pretrained weights.
            **kwargs: Additional keyword arguments.

        Returns:
            ProteinBERTModel instance.
        """
        return cls(model_path=model_path, **kwargs)
