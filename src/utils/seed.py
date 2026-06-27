"""Seeding Utilities for the Protein Foundation Model Benchmark Framework.

Provides functions for setting random seeds for reproducibility.
"""

import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)


_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed.
        deterministic: Whether to use deterministic algorithms.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def get_seed() -> Optional[int]:
    """Get current global seed.

    Returns:
        Current seed or None if not set.
    """
    return _GLOBAL_SEED


def seed_worker(worker_id: int) -> None:
    """Worker init function for DataLoader reproducibility.

    Args:
        worker_id: Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed: int = 42) -> torch.Generator:
    """Create PyTorch generator with seed.

    Args:
        seed: Random seed.

    Returns:
        Torch generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def set_torch_deterministic(enabled: bool = True) -> None:
    """Enable/disable deterministic algorithms in PyTorch.

    Args:
        enabled: Whether to enable deterministic mode.
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            pass


def get_rng_state() -> Dict[str, Any]:
    """Get current RNG states.

    Returns:
        Dictionary of RNG states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Set RNG states.

    Args:
        state: Dictionary of RNG states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])