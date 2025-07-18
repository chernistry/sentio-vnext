from __future__ import annotations

"""Mini cross-encoder reranker using sentence-transformers."""

import logging
from typing import List

from sentence_transformers import CrossEncoder  # type: ignore

from src.core.models.document import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:  # noqa: WPS110 – class name OK
    """Re-rank documents via a lightweight cross-encoder scoring model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        self._encoder = CrossEncoder(model_name, max_length=max_length)
        logger.info("CrossEncoderReranker loaded model '%s'", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:  # noqa: D401 – simple
        """Return *docs* ordered by relevance according to cross-encoder."""
        if not docs:
            return []

        pairs = [[query, d.text] for d in docs]
        scores = self._encoder.predict(pairs, convert_to_numpy=True)
        # attach scores
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda t: t[1], reverse=True)
        reranked = [d for d, _s in scored_docs[:top_k]]
        # store new scores in metadata
        for doc, score in scored_docs:
            doc.metadata["rerank_score"] = float(score)
        return reranked 