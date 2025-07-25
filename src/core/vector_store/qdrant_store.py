from __future__ import annotations

"""Qdrant vector-store wrapper.

This thin abstraction hides the concrete *qdrant-client* API and provides
some ergonomic helpers (collection bootstrap, health-check).

The implementation is intentionally minimal – it is *not* a full-blown
repository pattern – but rather a convenience layer that centralises
configuration and future-proofs against API shape changes.
"""

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.utils.settings import settings

__all__ = ["QdrantStore"]

logger = logging.getLogger(__name__)


def _get_default_timeout() -> float:
    return 10.0


@dataclass(slots=True)
class QdrantStore(VectorStore):  # noqa: WPS110 (data‐class is fine)
    """Wrapper around :pymod:`qdrant_client` with helper routines.

    The class supports Qdrant Cloud instances.
    *URL* and *API key* are resolved from environment variables unless
    provided explicitly.

    Note
    ----
    The sync client is used for now.  Async operations can be introduced
    later by swapping to :class:`qdrant_client.AsyncQdrantClient`.
    """

    collection_name: str
    vector_size: int = 1024
    distance: str = "Cosine"
    url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = field(default_factory=_get_default_timeout)
    embedding: Optional[Embeddings] = None
    content_payload_key: str = "content"
    metadata_payload_key: str = "metadata"

    _client: QdrantClient = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – simple
        """Initialise underlying Qdrant client and ensure collection exists."""

        # Resolve configuration lazily to make the class test-friendly.
        resolved_url = self.url or settings.qdrant_url
        if resolved_url is None:
            raise ValueError("QDRANT_URL env var or 'url' argument must be provided.")

        resolved_key = self.api_key or settings.qdrant_api_key
        if resolved_key is None:
            raise ValueError("QDRANT_API_KEY env var or 'api_key' argument must be provided.")

        # Ensure HTTPS for cloud instances
        if ".cloud.qdrant.io" in resolved_url:
            if not resolved_url.startswith("https://"):
                resolved_url = f"https://{resolved_url.removeprefix('http://')}"
                logger.info("Converted Qdrant cloud URL to HTTPS: %s", resolved_url)
        
        # For debugging
        logger.debug("Connecting to Qdrant at %s", resolved_url)

        # Instantiate client (sync for now).
        self._client = QdrantClient(url=resolved_url, api_key=resolved_key, timeout=self.timeout)

        logger.debug("Qdrant client initialised (url=%s, collection=%s)", resolved_url, self.collection_name)

        # Ensure collection exists.
        self._bootstrap_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def health_check(self) -> bool:  # noqa: D401 – simple
        """Return *True* when Qdrant instance is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception as exc:  # pragma: no cover – network exceptions vary
            logger.warning("Qdrant health-check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # LangChain VectorStore API Implementation
    # ------------------------------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts.
            **kwargs: Optional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        if self.embedding is None:
            raise ValueError("Embeddings must be provided for adding texts.")

        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            embeddings: List of embeddings to add.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to associate with the texts.
            **kwargs: Optional keyword arguments.

        Returns:
            List of IDs of the added embeddings.
        """
        texts_list = list(texts)
        if len(texts_list) != len(embeddings):
            raise ValueError("Number of texts and embeddings must be the same.")

        metadatas = metadatas or [{}] * len(texts_list)
        if len(metadatas) != len(texts_list):
            raise ValueError("Number of texts and metadatas must be the same.")

        points = []
        ids_out = []

        for i, (text, embedding, metadata) in enumerate(zip(texts_list, embeddings, metadatas)):
            point_id = str(ids[i]) if ids and i < len(ids) else str(i)
            point = rest.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    self.content_payload_key: text,
                    self.metadata_payload_key: metadata,
                },
            )
            points.append(point)
            ids_out.append(point_id)

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return ids_out

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query with relevance scores.

        Args:
            query: Text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments.

        Returns:
            List of tuples of (document, relevance score).
        """
        if self.embedding is None:
            raise ValueError("Embeddings must be provided for similarity search.")

        embedding = self.embedding.embed_query(query)
        search_params = kwargs.get("search_params", None)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k,
            query_filter=self._convert_filter(filter) if filter else None,
            search_params=search_params,
        )

        documents = []
        for result in results:
            payload = result.payload
            text = payload.get(self.content_payload_key, "")
            metadata = payload.get(self.metadata_payload_key, {})
            documents.append((Document(page_content=text, metadata=metadata), result.score))

        return documents

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to query.

        Args:
            query: Text to search for.
            k: Number of results to return.
            filter: Filter by metadata.
            **kwargs: Additional arguments.

        Returns:
            List of documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def delete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional arguments.

        Returns:
            Boolean indicating whether the operation was successful.
        """
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(
                points=ids,
            ),
        )
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _bootstrap_collection(self) -> None:  # noqa: D401 – simple
        """Create the collection if it does not yet exist."""
        collections = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in collections:
            return  # ✅ Already exists

        logger.info("Creating Qdrant collection '%s' (vector size=%s, distance=%s)", self.collection_name, self.vector_size, self.distance)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(  # type: ignore[arg-type]
                size=self.vector_size,
                distance=rest.Distance[self.distance.upper()],
            ),
        )

    def _convert_filter(self, filter_dict: Dict[str, Any]) -> Any:
        """Convert LangChain filter dict to Qdrant filter."""
        # Simple implementation for basic filters
        filter_clauses = []
        for key, value in filter_dict.items():
            metadata_key = f"{self.metadata_payload_key}.{key}"
            filter_clauses.append(rest.FieldCondition(
                key=metadata_key,
                match=rest.MatchValue(value=value),
            ))

        if len(filter_clauses) == 1:
            return filter_clauses[0]
        return rest.Filter(
            must=filter_clauses
        )

    # ------------------------------------------------------------------
    # Convenience passthroughs
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:  # noqa: D401 – simple
        """Delegate unknown attributes to the underlying client."""
        return getattr(self._client, item)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "documents",
        vector_size: Optional[int] = None,
        **kwargs: Any,
    ) -> "QdrantStore":
        """Return VectorStore instance from texts.

        Args:
            texts: List of texts.
            embedding: Embeddings to use.
            metadatas: Optional list of metadatas.
            ids: Optional list of IDs.
            collection_name: Name of the collection to store the vectors in.
            vector_size: Size of the embedding vectors.
            **kwargs: Additional arguments.

        Returns:
            A QdrantStore instance.
        """
        # Determine vector size if not provided
        if vector_size is None:
            if texts:
                # Embed the first text to get the vector size
                vector_size = len(embedding.embed_query(texts[0]))
            else:
                vector_size = 1024  # Default size

        instance = cls(
            collection_name=collection_name,
            vector_size=vector_size,
            embedding=embedding,
            **kwargs,
        )

        if texts:
            instance.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return instance 