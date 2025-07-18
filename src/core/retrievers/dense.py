from __future__ import annotations

"""Dense similarity search via Qdrant."""

import logging
from typing import List, Optional

from qdrant_client import QdrantClient

from core.embeddings.base import BaseEmbedder
from core.models.document import Document
from core.vector_store.qdrant_store import QdrantStore
from .base import BaseRetriever

logger = logging.getLogger(__name__)

# Default vector name for Qdrant points – overridable via env var
_DEFAULT_VECTOR_NAME = "text-dense"


class DenseRetriever(BaseRetriever):  # noqa: WPS110 – name is fine
    """Dense vector retriever powered by Qdrant."""

    def __init__(
        self,
        client: QdrantClient | QdrantStore,
        embedder: BaseEmbedder,
        collection_name: str,
        vector_name: str | None = None,
    ) -> None:
        self._client: QdrantClient = (
            client._client if isinstance(client, QdrantStore) else client  # type: ignore[attr-defined]
        )
        self._embedder = embedder
        self._collection = collection_name
        self._vector_name = vector_name or _DEFAULT_VECTOR_NAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:  # noqa: D401 – simple
        logger.debug("DenseRetriever: searching for '%s' (top_k=%s)", query, top_k)
        query_vec = self._embedder.embed_sync(query)
        points = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            vector_name=self._vector_name,
        )
        docs: List[Document] = []
        for p in points:
            text = (p.payload or {}).get("text") if isinstance(p.payload, dict) else None
            docs.append(
                Document(
                    id=str(p.id),
                    text=text or "",
                    metadata={"score": p.score, **(p.payload or {})},
                )
            )
        return docs 