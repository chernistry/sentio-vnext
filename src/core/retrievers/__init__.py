from __future__ import annotations

"""Retriever package public surface."""

from importlib import import_module
from typing import Any

__all__ = [
    "get_retriever",
]


def get_retriever(kind: str, **kwargs: Any):  # noqa: D401 – simple
    """Return retriever instance by *kind* identifier.

    Args:
        kind: Lower-case identifier (``"dense"``, ``"hybrid"``).
        **kwargs: Forwarded to concrete constructor.
    """

    kind = kind.lower()
    if kind == "dense":
        module = import_module("core.retrievers.dense")
        return module.DenseRetriever(**kwargs)
    if kind == "hybrid":
        module = import_module("core.retrievers.hybrid")
        return module.HybridRetriever(**kwargs)

    raise ValueError(f"Unknown retriever kind: {kind}")
