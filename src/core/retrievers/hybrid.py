from __future__ import annotations

"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict
import logging
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi  # Lightweight BM25 implementation

from src.core.models.document import Document
from .base import BaseRetriever
from .dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):  # noqa: WPS110
    """Combine dense retrieval with lexical BM25 using RRF fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        corpus_docs: Optional[List[Document]] = None,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense_retriever
        self._rrf_k = rrf_k

        # Build BM25 index if corpus provided
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        if corpus_docs:
            self._build_sparse_index(corpus_docs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:  # noqa: D401 – simple
        dense_hits = self._dense.retrieve(query, top_k=top_k)
        sparse_hits: List[Tuple[str, float]] = []

        if self._bm25 is not None:
            tokens = query.split()
            scores = self._bm25.get_scores(tokens)
            # Pair doc_id with score and take top_k
            scored = list(zip(self._doc_ids, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            sparse_hits = scored[:top_k]

        fused_scores = defaultdict(float)

        # Dense ranking
        for rank, doc in enumerate(dense_hits):
            fused_scores[doc.id] += 1 / (self._rrf_k + rank)

        # Sparse ranking
        for rank, (doc_id, _score) in enumerate(sparse_hits):
            fused_scores[doc_id] += 1 / (self._rrf_k + rank)

        # Build final docs map from dense docs + corpus docs
        id_to_doc: dict[str, Document] = {doc.id: doc for doc in dense_hits}

        # If sparse results include docs not in dense hits, fetch from corpus index
        if self._bm25 is not None:
            for doc_id in self._doc_ids:
                if doc_id not in id_to_doc:
                    idx = self._doc_ids.index(doc_id)
                    if idx < len(self._bm25.corpus):
                        text = " ".join(self._bm25.corpus[idx])
                        id_to_doc[doc_id] = Document(id=doc_id, text=text)

        # Sort by fused score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [id_to_doc[item[0]] for item in ranked]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_sparse_index(self, docs: List[Document]) -> None:  # noqa: D401 – simple
        self._doc_ids = [doc.id for doc in docs]
        tokenised = [doc.text.split() for doc in docs]
        if tokenised:
            self._bm25 = BM25Okapi(tokenised)
        logger.info("HybridRetriever: built BM25 index on %s docs", len(self._doc_ids)) 