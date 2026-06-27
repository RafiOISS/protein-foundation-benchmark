"""TrainerFactory — returns the correct trainer for a given model type."""

from typing import Optional

from .base_trainer import BaseTrainer
from .torch_trainer import TorchTrainer
from ..framework.checkpoint import Checkpoint
from ..utils.logging import get_logger


logger = get_logger(__name__)


def create_trainer(
    model_type: str,
    checkpoint: Optional[Checkpoint] = None,
    device: str = "auto",
) -> BaseTrainer:
    """Factory: returns a TorchTrainer or TensorFlowTrainer based on model_type.

    Args:
        model_type: 'esm2', 'protbert', 'prott5', 'cnn', 'bilstm', 'proteinbert'.
        checkpoint: Optional Checkpoint instance.
        device: Device string.

    Returns:
        A trainer instance.
    """
    if model_type == "proteinbert":
        from .tensorflow_trainer import TensorFlowTrainer
        logger.info("Creating TensorFlowTrainer for ProteinBERT")
        return TensorFlowTrainer()

    logger.info(f"Creating TorchTrainer for {model_type}")
    return TorchTrainer(checkpoint=checkpoint, device=device)