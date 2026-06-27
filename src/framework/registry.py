"""Core registry — shared base for all registries."""

from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry pattern used as a base for Model, Dataset, and Metric registries."""

    def __init__(self, name: str = "registry") -> None:
        self._name = name
        self._items: Dict[str, Type[T]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def register(self, key: str, item: Type[T], config: Optional[Dict[str, Any]] = None) -> None:
        if key in self._items:
            raise KeyError(f"'{key}' already registered in {self._name}")
        self._items[key] = item
        if config:
            self._configs[key] = config

    def get(self, key: str) -> Type[T]:
        if key not in self._items:
            raise KeyError(f"'{key}' not found in {self._name}. Available: {list(self._items.keys())}")
        return self._items[key]

    def list(self) -> List[str]:
        return list(self._items.keys())

    def get_config(self, key: str) -> Dict[str, Any]:
        return self._configs.get(key, {}).copy()

    def unregister(self, key: str) -> None:
        self._items.pop(key, None)
        self._configs.pop(key, None)