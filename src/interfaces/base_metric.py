"""Base metric interface for the Protein Foundation Model Benchmark Framework.

All evaluation metrics must inherit from BaseMetric.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    Provides a unified interface for computing metrics across task types.
    """

    @abstractmethod
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute metric(s).

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional parameters.

        Returns:
            Dictionary of metric name -> value.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass

    @abstractmethod
    def task_types(self) -> List[str]:
        """Return supported task types."""
        pass

    @abstractmethod
    def requires_probabilities(self) -> bool:
        """Whether this metric needs probabilities instead of logits."""
        pass