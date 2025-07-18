from __future__ import annotations

"""Reranker package providing quality-aware reordering of retrieved docs."""

from importlib import import_module
from typing import Any

__all__ = ["get_reranker"]


def get_reranker(kind: str | None = None, **kwargs: Any):  # noqa: D401 – simple
    """Return a reranker instance.

    Currently supports only ``"cross-encoder"`` (default).
    """

    kind = (kind or "cross-encoder").lower()
    if kind in {"cross", "cross-encoder", "mini-cross"}:
        module = import_module("core.rerankers.cross_encoder")
        return module.CrossEncoderReranker(**kwargs)

    raise ValueError(f"Unknown reranker kind: {kind}")
