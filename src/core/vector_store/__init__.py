from __future__ import annotations

"""Vector store layer abstraction for Sentio.

Currently exposes a thin wrapper around Qdrant Cloud.  In future this
module may grow to support additional providers (e.g. Weaviate, Milvus).
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover – import-time only
    from .qdrant_store import QdrantStore  # noqa: F401  


def get_vector_store(name: str, **kwargs: Any):  # noqa: D401 – simple
    """Factory returning the requested vector store implementation.

    Args:
        name: Lower-case identifier (e.g. ``"qdrant"``).
        **kwargs: Forwarded to the concrete class constructor.

    Returns:
        Instantiated vector store client.

    Raises:
        ValueError: If *name* is unknown or the backend is not installed.
    """

    name = name.lower()
    if name == "qdrant":
        module = import_module("src.core.vector_store.qdrant_store")
        return module.QdrantStore(**kwargs)

    raise ValueError(f"Unknown vector store backend: {name}") 