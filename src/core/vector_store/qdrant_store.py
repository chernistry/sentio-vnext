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
from typing import Any, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from utils.settings import settings

__all__ = ["QdrantStore"]

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


@dataclass(slots=True)
class QdrantStore:  # noqa: WPS110 (data‐class is fine)
    """Wrapper around :pymod:`qdrant_client` with helper routines.

    The class supports both public Qdrant Cloud and self-hosted instances.
    *URL* and *API key* are resolved from environment variables unless
    provided explicitly.

    Note
    ----
    The sync client is used for now.  Async operations can be introduced
    later by swapping to :class:`qdrant_client.AsyncQdrantClient`.
    """

    collection_name: str
    vector_size: int
    distance: str = "Cosine"
    url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: httpx.Timeout = field(default=DEFAULT_TIMEOUT)

    _client: QdrantClient = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – simple
        """Initialise underlying Qdrant client and ensure collection exists."""

        # Resolve configuration lazily to make the class test-friendly.
        resolved_url = self.url or getattr(settings, "qdrant_url", None) or os.getenv("QDRANT_URL")
        if resolved_url is None:
            raise ValueError("QDRANT_URL env var or 'url' argument must be provided.")

        resolved_key = self.api_key or getattr(settings, "qdrant_api_key", None) or os.getenv("QDRANT_API_KEY")

        # Instantiate client (sync for now).
        self._client = QdrantClient(url=resolved_url, api_key=resolved_key, timeout=self.timeout)  # type: ignore[arg-type]

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

    # ------------------------------------------------------------------
    # Convenience passthroughs
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:  # noqa: D401 – simple
        """Delegate unknown attributes to the underlying client."""
        return getattr(self._client, item) 