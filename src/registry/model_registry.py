"""Model registry for the Protein Foundation Model Benchmark Framework.

Allows models to be registered by name and instantiated dynamically.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces.base_model import BaseProteinModel
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ModelRegistry:
    """Registry for protein foundation models.

    Models register themselves with a name and are instantiated on demand.
    """

    _models: Dict[str, Type[BaseProteinModel]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseProteinModel],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model class.

        Args:
            name: Unique model identifier (e.g., 'esm2', 'protbert').
            model_class: Model class (must subclass BaseProteinModel).
            config: Optional default configuration.
        """
        if not issubclass(model_class, BaseProteinModel):
            raise TypeError(f"{model_class.__name__} must subclass BaseProteinModel")

        cls._models[name] = model_class
        if config:
            cls._configs[name] = config

        logger.info(f"Registered model '{name}' -> {model_class.__name__}")

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseProteinModel:
        """Create a model instance by name.

        Args:
            name: Registered model name.
            config: Model configuration (merged with defaults).
            **kwargs: Additional keyword arguments passed to constructor.

        Returns:
            Model instance.

        Raises:
            ValueError: If model name is not registered.
        """
        if name not in cls._models:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {list(cls._models.keys())}"
            )

        model_class = cls._models[name]
        merged_config = {**cls._configs.get(name, {}), **(config or {})}

        logger.info(f"Creating model '{name}' ({model_class.__name__})")
        return model_class(config=merged_config, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[BaseProteinModel]:
        """Get the model class for a registered name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model '{name}'")
        return cls._models[name]

    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """Get default config for a registered model."""
        return cls._configs.get(name, {}).copy()

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a model."""
        cls._models.pop(name, None)
        cls._configs.pop(name, None)
        logger.info(f"Unregistered model '{name}'")