from __future__ import annotations

"""Common interface for document retrievers."""

import abc
from typing import List, Protocol, Sequence

from core.models.document import Document

__all__ = ["BaseRetriever"]


class BaseRetriever(abc.ABC):
    """Abstract base class for document retrievers."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:  # noqa: D401 – simple
        """Return top_k relevant documents."""
        raise NotImplementedError

    async def retrieve_async(self, query: str, top_k: int = 10) -> List[Document]:  # noqa: D401
        """Async wrapper for :py:meth:`retrieve` running in executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, top_k)) 