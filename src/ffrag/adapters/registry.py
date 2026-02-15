from __future__ import annotations

from collections.abc import Callable

from .base import BaseAdapter


AdapterFactory = Callable[..., BaseAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("adapter name is required")
    _REGISTRY[key] = factory


def get_adapter(name: str) -> AdapterFactory:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"adapter not registered: {name}")
    return _REGISTRY[key]


def list_adapters() -> list[str]:
    return sorted(_REGISTRY.keys())
