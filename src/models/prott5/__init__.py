"""ProtT5 Model Wrapper for the Protein Foundation Model Benchmark Framework.

Wrapper for ProtT5 models (Rostlab/prot_t5_*) using HuggingFace Transformers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class ProtT5(BaseProteinModel):
    """Wrapper for ProtT5 models from Rostlab.

    ProtT5 is a T5-based protein language model trained on UniRef50/BFD.
    Uses the encoder part of T5 for embedding extraction.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize ProtT5.

        Args:
            model_name: HuggingFace model identifier.
            config: Model configuration.
            device: Device to run on.
            **kwargs: Additional arguments.
        """
        config = config or {}
        config.setdefault("model_type", "prott5")
        config.setdefault("huggingface_id", model_name)
        config.setdefault("max_seq_len", 1022)
        config.setdefault("embedding_dim", 1024)

        super().__init__(config, device)
        self.model_name = model_name
        self._model: Optional[T5EncoderModel] = None
        self._tokenizer: Optional[T5Tokenizer] = None

        logger.info(f"ProtT5 wrapper initialized for {model_name}")

    def _load_model(self) -> None:
        """Load the ProtT5 encoder model and tokenizer."""
        try:
            self._model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self._model.eval()
            logger.info(f"Loaded ProtT5 from {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load ProtT5: {e}")
            raise

    def _load_tokenizer(self) -> None:
        """Load tokenizer if not already loaded."""
        if self._tokenizer is None:
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through ProtT5 encoder.

        Args:
            sequences: List of protein sequences.

        Returns:
            Last hidden states of shape (batch_size, seq_len, hidden_dim).
        """
        if self._model is None:
            self._load_model()

        # ProtT5 expects space-separated amino acids
        processed_sequences = [" ".join(list(seq)) for seq in sequences]

        inputs = self._tokenizer(
            processed_sequences,
            max_length=self.config.get("max_seq_len", 1022),
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        return outputs.last_hidden_state

    def extract_embeddings(
        self,
        sequences: List[str],
        layers: Union[str, List[int]] = "last",
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract embeddings from ProtT5 encoder.

        Args:
            sequences: List of protein sequences.
            layers: Layers to extract (T5 encoder only has last layer output).
            pooling: Pooling strategy ('mean', 'cls', 'max').

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        if self._model is None:
            self._load_model()

        processed_sequences = [" ".join(list(seq)) for seq in sequences]

        inputs = self._tokenizer(
            processed_sequences,
            max_length=self.config.get("max_seq_len", 1022),
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        layer_embeddings = outputs.last_hidden_state

        if pooling == "cls":
            embeddings = layer_embeddings[:, 0, :]
        elif pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (layer_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        elif pooling == "max":
            embeddings = layer_embeddings.max(1).values
        else:
            embeddings = layer_embeddings.mean(1)

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 1024)

    def get_max_seq_len(self) -> int:
        return self.config.get("max_seq_len", 1022)

    def get_num_layers(self) -> int:
        return 24  # ProtT5-XL has 24 encoder layers

    def get_tokenizer(self) -> T5Tokenizer:
        """Get the tokenizer."""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        logger.info(f"Saved ProtT5 to {path}")

    @classmethod
    def from_pretrained(cls, model_name: str = "Rostlab/prot_t5_xl_uniref50", **kwargs) -> "ProtT5":
        """Load pretrained ProtT5."""
        return cls(model_name=model_name, **kwargs)