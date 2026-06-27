"""BiLSTM Baseline Model for the Protein Foundation Model Benchmark Framework.

Simple BiLSTM baseline for protein sequence classification/regression.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class BiLSTMBaseline(BaseProteinModel):
    """Simple BiLSTM baseline for protein sequences.

    Uses bidirectional LSTM with character-level embeddings for sequence encoding.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize BiLSTM baseline.

        Args:
            config: Model configuration.
            device: Device to run on.
            **kwargs: Additional arguments.
        """
        config = config or {}
        config.setdefault("model_type", "bilstm")
        config.setdefault("vocab_size", 25)  # 20 AAs + special tokens
        config.setdefault("embed_dim", 128)
        config.setdefault("hidden_dim", 256)
        config.setdefault("num_layers", 2)
        config.setdefault("max_seq_len", 1024)
        config.setdefault("embedding_dim", 512)
        config.setdefault("num_classes", 1)
        config.setdefault("dropout", 0.1)
        config.setdefault("bidirectional", True)
        config.setdefault("pooling", "last")

        super().__init__(config, device)

        self.vocab_size = config["vocab_size"]
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.max_seq_len = config["max_seq_len"]
        self.embedding_dim = config["embedding_dim"]
        self.num_classes = config["num_classes"]
        self.dropout_rate = config["dropout"]
        self.bidirectional = config["bidirectional"]
        self.pooling = config["pooling"]

        self._build_model()

        logger.info(f"BiLSTMBaseline initialized with embedding_dim={self.embedding_dim}")

    def _build_model(self) -> None:
        """Build the BiLSTM model architecture."""
        # Character embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )

        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        # Pooling and classifier
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, self.num_classes)

        self.to(self.device)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through BiLSTM.

        Args:
            sequences: List of protein sequences.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self._sequences_to_tensor(sequences).to(self.device)

        # Embedding
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_dim * num_directions)

        # Pooling
        if self.pooling == "last":
            # Use last hidden state (concatenated for bidirectional)
            if self.bidirectional:
                embeddings = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                embeddings = h_n[-1]
        elif self.pooling == "mean":
            embeddings = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            embeddings = lstm_out.max(dim=1).values
        else:
            embeddings = lstm_out[:, -1, :]

        # Classifier
        embeddings = self.dropout(embeddings)
        embeddings = F.relu(self.fc(embeddings))
        embeddings = self.dropout(embeddings)
        logits = self.output_layer(embeddings)

        return logits

    def _sequences_to_tensor(self, sequences: List[str]) -> torch.Tensor:
        """Convert protein sequences to token indices.

        Args:
            sequences: List of protein sequences.

        Returns:
            Tensor of shape (batch_size, max_seq_len).
        """
        aa_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_to_idx["X"] = 21
        aa_to_idx["<pad>"] = 0
        aa_to_idx["<cls>"] = 22
        aa_to_idx["<sep>"] = 23
        aa_to_idx["<mask>"] = 24

        batch = []
        for seq in sequences:
            seq = seq[:self.max_seq_len - 2]
            tokens = [aa_to_idx["<cls>"]] + [aa_to_idx.get(aa, aa_to_idx["X"]) for aa in seq] + [aa_to_idx["<sep>"]]
            if len(tokens) < self.max_seq_len:
                tokens += [aa_to_idx["<pad>"]] * (self.max_seq_len - len(tokens))
            else:
                tokens = tokens[:self.max_seq_len]
            batch.append(tokens)

        return torch.tensor(batch, dtype=torch.long)

    def extract_embeddings(
        self,
        sequences: List[str],
        layers: Union[str, List[int]] = "last",
        pooling: str = "last",
    ) -> torch.Tensor:
        """Extract embeddings from BiLSTM (penultimate layer).

        Args:
            sequences: List of protein sequences.
            layers: Ignored for BiLSTM.
            pooling: Pooling strategy.

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        x = self._sequences_to_tensor(sequences).to(self.device)
        x = self.embedding(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        if pooling == "last":
            if self.bidirectional:
                embeddings = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                embeddings = h_n[-1]
        elif pooling == "mean":
            embeddings = lstm_out.mean(dim=1)
        elif pooling == "max":
            embeddings = lstm_out.max(dim=1).values
        else:
            embeddings = lstm_out[:, -1, :]

        embeddings = self.dropout(embeddings)
        embeddings = F.relu(self.fc(embeddings))

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_num_layers(self) -> int:
        return self.num_layers

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved BiLSTMBaseline to {path}")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], config: Dict[str, Any], **kwargs) -> "BiLSTMBaseline":
        """Load pretrained BiLSTMBaseline."""
        model = cls(config=config, **kwargs)
        model.load_state_dict(torch.load(path, map_location=model.device))
        return model