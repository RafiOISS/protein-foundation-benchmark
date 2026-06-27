"""Metric registry for the Protein Foundation Model Benchmark Framework.

Allows metrics to be registered by name and computed dynamically.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ..interfaces.base_metric import BaseMetric
from ..utils.logging import get_logger


logger = get_logger(__name__)


_DefaultMetricFn = Callable[[torch.Tensor, torch.Tensor], float]


class MetricRegistry:
    """Registry for evaluation metrics.

    Supports both class-based (BaseMetric subclass) and function-based metrics.
    """

    _metrics: Dict[str, Union[type, _DefaultMetricFn]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        metric: Union[type, _DefaultMetricFn],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a metric.

        Args:
            name: Unique metric name (e.g., 'accuracy', 'spearman').
            metric: BaseMetric subclass or callable(predictions, targets) -> float.
            config: Optional default configuration.
        """
        if isinstance(metric, type) and issubclass(metric, BaseMetric):
            pass
        elif callable(metric):
            pass
        else:
            raise TypeError("Metric must be a BaseMetric subclass or callable")

        cls._metrics[name] = metric
        if config:
            cls._configs[name] = config

        logger.info(f"Registered metric '{name}'")

    @classmethod
    def compute(
        cls,
        name: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute a registered metric.

        Args:
            name: Registered metric name.
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments passed to the metric.

        Returns:
            Metric value.
        """
        if name not in cls._metrics:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Available: {list(cls._metrics.keys())}"
            )

        metric = cls._metrics[name]
        merged_config = {**cls._configs.get(name, {}), **kwargs}

        if isinstance(metric, type) and issubclass(metric, BaseMetric):
            return metric(**merged_config)(predictions, targets)
        return metric(predictions, targets, **merged_config)

    @classmethod
    def compute_all(
        cls,
        metrics: List[str],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute multiple metrics at once.

        Args:
            metrics: List of metric names.
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of metric name -> value.
        """
        results = {}
        for name in metrics:
            try:
                results[name] = cls.compute(name, predictions, targets, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to compute metric '{name}': {e}")
                results[name] = float("nan")
        return results

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a metric."""
        cls._metrics.pop(name, None)
        cls._configs.pop(name, None)
        logger.info(f"Unregistered metric '{name}'")