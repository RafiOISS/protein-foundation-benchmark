"""CNN Baseline Model for the Protein Foundation Model Benchmark Framework.

Simple CNN baseline for protein sequence classification/regression.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logging import get_logger
from ...interfaces.base_model import BaseProteinModel


logger = get_logger(__name__)


class CNNBaseline(BaseProteinModel):
    """Simple CNN baseline for protein sequences.

    Uses character-level CNN with multiple filter sizes for sequence encoding.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize CNN baseline.

        Args:
            config: Model configuration.
            device: Device to run on.
            **kwargs: Additional arguments.
        """
        config = config or {}
        config.setdefault("model_type", "cnn")
        config.setdefault("vocab_size", 25)  # 20 AAs + special tokens
        config.setdefault("embed_dim", 128)
        config.setdefault("num_filters", [64, 128, 256, 512])
        config.setdefault("kernel_sizes", [3, 5, 7, 9])
        config.setdefault("max_seq_len", 1024)
        config.setdefault("embedding_dim", 512)
        config.setdefault("num_classes", 1)
        config.setdefault("dropout", 0.1)
        config.setdefault("pooling", "global_max")

        super().__init__(config, device)

        self.vocab_size = config["vocab_size"]
        self.embed_dim = config["embed_dim"]
        self.num_filters = config["num_filters"]
        self.kernel_sizes = config["kernel_sizes"]
        self.max_seq_len = config["max_seq_len"]
        self.embedding_dim = config["embedding_dim"]
        self.num_classes = config["num_classes"]
        self.dropout_rate = config["dropout"]
        self.pooling = config["pooling"]

        self._build_model()

        logger.info(f"CNNBaseline initialized with embedding_dim={self.embedding_dim}")

    def _build_model(self) -> None:
        """Build the CNN model architecture."""
        # Character embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=nf,
                kernel_size=ks,
                padding=ks // 2,
            )
            for nf, ks in zip(self.num_filters, self.kernel_sizes)
        ])

        # Batch norm for each conv
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(nf) for nf in self.num_filters
        ])

        # Global pooling
        if self.pooling == "global_max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif self.pooling == "global_avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier head
        total_filters = sum(self.num_filters)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(total_filters, self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, self.num_classes)

        self.to(self.device)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass through CNN.

        Args:
            sequences: List of protein sequences.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Convert sequences to token indices
        x = self._sequences_to_tensor(sequences)
        x = x.to(self.device)

        # Embedding: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = self.embedding(x).transpose(1, 2)

        # Convolutions + pooling
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            out = F.relu(bn(conv(x)))
            out = self.pool(out).squeeze(-1)
            conv_outputs.append(out)

        # Concatenate
        x = torch.cat(conv_outputs, dim=1)

        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.output_layer(x)

        return logits

    def _sequences_to_tensor(self, sequences: List[str]) -> torch.Tensor:
        """Convert protein sequences to token indices.

        Args:
            sequences: List of protein sequences.

        Returns:
            Tensor of shape (batch_size, max_seq_len).
        """
        # Simple character-level encoding
        aa_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_to_idx["X"] = 21  # Unknown
        aa_to_idx["<pad>"] = 0
        aa_to_idx["<cls>"] = 22
        aa_to_idx["<sep>"] = 23
        aa_to_idx["<mask>"] = 24

        batch = []
        for seq in sequences:
            # Truncate
            seq = seq[:self.max_seq_len - 2]
            # Add special tokens
            tokens = [aa_to_idx["<cls>"]] + [aa_to_idx.get(aa, aa_to_idx["X"]) for aa in seq] + [aa_to_idx["<sep>"]]
            # Pad
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
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract embeddings from CNN (penultimate layer).

        Args:
            sequences: List of protein sequences.
            layers: Ignored for CNN.
            pooling: Pooling strategy.

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim).
        """
        x = self._sequences_to_tensor(sequences).to(self.device)
        x = self.embedding(x).transpose(1, 2)

        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            out = F.relu(bn(conv(x)))
            out = self.pool(out).squeeze(-1)
            conv_outputs.append(out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        embeddings = F.relu(self.fc(x))

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_num_layers(self) -> int:
        return len(self.convs)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved CNNBaseline to {path}")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], config: Dict[str, Any], **kwargs) -> "CNNBaseline":
        """Load pretrained CNNBaseline."""
        model = cls(config=config, **kwargs)
        model.load_state_dict(torch.load(path, map_location=model.device))
        return model