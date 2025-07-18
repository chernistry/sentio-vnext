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
        module = import_module("src.core.retrievers.dense")
        return module.DenseRetriever(**kwargs)
    if kind == "hybrid":
        module = import_module("src.core.retrievers.hybrid")
        # Extract top_k from kwargs as it's not a parameter for HybridRetriever constructor
        kwargs.pop("top_k", 10)
        # In a test environment, mock objects might be passed directly
        if "dense_retriever" not in kwargs:
            # For actual implementation, the dense retriever should be created externally
            # and passed explicitly as the tests expect the factory function to be mocked
            raise ValueError(
                "For hybrid retriever, 'dense_retriever' must be provided explicitly"
            )
        return module.HybridRetriever(**kwargs)

    raise ValueError(f"Unknown retriever kind: {kind}")
