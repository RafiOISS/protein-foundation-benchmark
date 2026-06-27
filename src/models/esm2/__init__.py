"""ESM-2 Model Wrapper for the Protein Foundation Model Benchmark Framework.

Wrapper for ESM-2 models (facebook/esm2_*) using HuggingFace Transformers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class ESM2(BaseProteinModel):
    """Wrapper for ESM-2 models from Facebook/Meta.

    ESM-2 is a family of transformer protein language models trained on UniRef50.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize ESM-2.

        Args:
            model_name: HuggingFace model identifier.
            config: Model configuration.
            device: Device to run on.
            **kwargs: Additional arguments.
        """
        config = config or {}
        config.setdefault("model_type", "esm2")
        config.setdefault("huggingface_id", model_name)
        config.setdefault("max_seq_len", 1022)
        config.setdefault("embedding_dim", 320)

        # Parse embedding dim from model name if possible
        if "8M" in model_name:
            config["embedding_dim"] = 320
        elif "35M" in model_name:
            config["embedding_dim"] = 480
        elif "150M" in model_name:
            config["embedding_dim"] = 640
        elif "650M" in model_name:
            config["embedding_dim"] = 1280
        elif "3B" in model_name:
            config["embedding_dim"] = 2560
        elif "15B" in model_name:
            config["embedding_dim"] = 5120

        super().__init__(config, device)
        self.model_name = model_name
        self._model: Optional[EsmModel] = None
        self._tokenizer: Optional[EsmTokenizer] = None

        logger.info(f"ESM-2 wrapper initialized for {model_name}")

    def _load_model(self) -> None:
        """Load the ESM-2 model and tokenizer."""
        try:
            self._model = EsmModel.from_pretrained(self.model_name).to(self.device)
            self._tokenizer = EsmTokenizer.from_pretrained(self.model_name)
            self._model.eval()
            logger.info(f"Loaded ESM-2 from {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load ESM-2: {e}")
            raise

    def _load_tokenizer(self) -> None:
        """Load tokenizer if not already loaded."""
        if self._tokenizer is None:
            self._tokenizer = EsmTokenizer.from_pretrained(self.model_name)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through ESM-2.

        Args:
            sequences: List of protein sequences.

        Returns:
            Last hidden states of shape (batch_size, seq_len, hidden_dim).
        """
        if self._model is None:
            self._load_model()

        inputs = self._tokenizer(
            sequences,
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
        """Extract embeddings from ESM-2.

        Args:
            sequences: List of protein sequences.
            layers: Layers to extract ('last', 'all', or list of indices).
            pooling: Pooling strategy ('mean', 'cls', 'max').

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        if self._model is None:
            self._load_model()

        inputs = self._tokenizer(
            sequences,
            max_length=self.config.get("max_seq_len", 1022),
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        if layers == "last":
            layer_embeddings = hidden_states[-1]
        elif layers == "all":
            layer_embeddings = torch.stack(hidden_states).mean(0)
        elif isinstance(layers, list):
            layer_embeddings = torch.stack([hidden_states[i] for i in layers]).mean(0)
        else:
            layer_embeddings = hidden_states[-1]

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
        return self.config.get("embedding_dim", 320)

    def get_max_seq_len(self) -> int:
        return self.config.get("max_seq_len", 1022)

    def get_num_layers(self) -> int:
        # Parse from model name
        if "t6" in self.model_name:
            return 6
        elif "t12" in self.model_name:
            return 12
        elif "t30" in self.model_name:
            return 30
        elif "t33" in self.model_name:
            return 33
        elif "t36" in self.model_name:
            return 36
        elif "t48" in self.model_name:
            return 48
        return 6

    def get_tokenizer(self) -> EsmTokenizer:
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
        logger.info(f"Saved ESM-2 to {path}")

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/esm2_t6_8M_UR50D", **kwargs) -> "ESM2":
        """Load pretrained ESM-2."""
        return cls(model_name=model_name, **kwargs)