"""ProteinBERT Model Wrapper for the Protein Foundation Model Benchmark Framework.

Wrapper for ProteinBERT model (requires TensorFlow).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class ProteinBERT(BaseProteinModel):
    """Wrapper for ProteinBERT model.

    ProteinBERT is a BERT-based protein language model that requires TensorFlow.
    This wrapper provides a PyTorch-compatible interface.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize ProteinBERT.

        Args:
            model_path: Path to pretrained ProteinBERT weights.
            config: Model configuration.
            device: Device to run on.
            **kwargs: Additional arguments.
        """
        config = config or {}
        config.setdefault("model_type", "proteinbert")
        config.setdefault("max_seq_len", 510)
        config.setdefault("embedding_dim", 768)

        super().__init__(config, device)
        self.model_path = Path(model_path) if model_path else None
        self._tf_model = None
        self._tokenizer = None

        logger.info("ProteinBERT wrapper initialized (TensorFlow backend required)")

    def _load_model(self) -> None:
        """Load the ProteinBERT model."""
        try:
            import tensorflow as tf
            from proteinbert import load_pretrained_model

            if self.model_path and self.model_path.exists():
                self._tf_model = load_pretrained_model(str(self.model_path))
            else:
                logger.warning("No ProteinBERT model path provided. Using random initialization.")
                self._tf_model = load_pretrained_model()

            logger.info("ProteinBERT model loaded successfully")
        except ImportError:
            logger.error("TensorFlow and proteinbert package required for ProteinBERT")
            raise
        except Exception as e:
            logger.error(f"Failed to load ProteinBERT: {e}")
            raise

    def _load_tokenizer(self) -> None:
        """Load the ProteinBERT tokenizer."""
        try:
            from proteinbert import ProteinBertTokenizer
            self._tokenizer = ProteinBertTokenizer()
            logger.info("ProteinBERT tokenizer loaded")
        except ImportError:
            logger.error("proteinbert package required for tokenizer")
            raise

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through ProteinBERT.

        Args:
            sequences: List of protein sequences.

        Returns:
            Model outputs (logits or embeddings).
        """
        if self._tf_model is None:
            self._load_model()

        if self._tokenizer is None:
            self._load_tokenizer()

        # Tokenize sequences
        encoded = self._tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            max_length=self.config.get("max_seq_len", 510),
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        # Get embeddings from TensorFlow model
        import tensorflow as tf
        outputs = self._tf_model(encoded["input_ids"])

        # Convert to PyTorch tensor
        embeddings = torch.from_numpy(outputs.numpy())
        return embeddings

    def extract_embeddings(
        self,
        sequences: List[str],
        layers: Union[str, List[int]] = "last",
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract embeddings from ProteinBERT.

        Args:
            sequences: List of protein sequences.
            layers: Layers to extract (ignored for ProteinBERT).
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
        return self.config.get("max_seq_len", 510)

    def get_num_layers(self) -> int:
        return 12  # ProteinBERT has 12 layers

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model weights."""
        if self._tf_model is not None:
            self._tf_model.save_weights(str(path))
            logger.info(f"Saved ProteinBERT to {path}")

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs) -> "ProteinBERT":
        """Load pretrained ProteinBERT."""
        return cls(model_path=model_path, **kwargs)