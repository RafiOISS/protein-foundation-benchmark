"""ProteinBERT Model Package.

Provides ProteinBERTModel — a TensorFlow-backed protein language model
wrapper with lazy imports. Registered with ModelRegistry on import.

Requires:
  - tensorflow (imported lazily, never at module level)
  - proteinbert (imported lazily, never at module level)
"""

from ...utils.logging import get_logger
from ...registry.model_registry import ModelRegistry
from .wrapper import ProteinBERTModel


logger = get_logger(__name__)


# Backward-compatible alias
ProteinBERT = ProteinBERTModel


# Register with ModelRegistry
ModelRegistry.register("proteinbert", ProteinBERTModel)

logger.debug("Registered model: proteinbert")


__all__ = [
    "ProteinBERTModel",
    "ProteinBERT",
]
