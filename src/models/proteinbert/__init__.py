"""ProteinBERT Model Package.

Provides ProteinBERTModel — a TensorFlow-backed protein language model
wrapper with lazy imports. Registered with ModelRegistry on import.

Includes ProteinBERTDataAdapter — a framework-to-model tensor bridge
for converting preprocessed outputs into ProteinBERT-ready tensors.

Includes CheckpointManager — centralized checkpoint management.
Includes ProteinBERTTrainer — TensorFlow training engine.

Requires:
  - tensorflow (imported lazily, never at module level)
  - proteinbert (imported lazily, never at module level)
"""

from ...utils.logging import get_logger
from ...registry.model_registry import ModelRegistry
from .wrapper import ProteinBERTModel
from .adapter import ProteinBERTDataAdapter, AdapterMetadata, TensorValidator
from .runtime import Runtime
from .checkpoints import CheckpointManager
from .trainer import ProteinBERTTrainer, TrainingHistory


logger = get_logger(__name__)


# Backward-compatible alias
ProteinBERT = ProteinBERTModel


# Register with ModelRegistry
ModelRegistry.register("proteinbert", ProteinBERTModel)

logger.debug("Registered model: proteinbert")


__all__ = [
    "ProteinBERTModel",
    "ProteinBERT",
    "ProteinBERTDataAdapter",
    "AdapterMetadata",
    "TensorValidator",
    "Runtime",
    "CheckpointManager",
    "ProteinBERTTrainer",
    "TrainingHistory",
]
