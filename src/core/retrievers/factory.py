from __future__ import annotations

"""Factory for creating retrievers based on environment configuration."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient

from src.core.embeddings.base import BaseEmbedder
from src.core.models.document import Document
from src.core.retrievers.base import BaseRetriever, ScorerPlugin
from src.core.retrievers.dense import DenseRetriever
from src.core.retrievers.hybrid import HybridRetriever
from src.core.retrievers.sparse import BM25Retriever, PyseriniBM25Retriever
from src.core.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def create_retriever_from_env(
    client: Union[QdrantClient, QdrantStore],
    embedder: BaseEmbedder,
    collection_name: Optional[str] = None,
    corpus_docs: Optional[List[Document]] = None,
    scorer_plugins: Optional[List[ScorerPlugin]] = None,
) -> BaseRetriever:
    """Create a retriever based on environment configuration.
    
    Args:
        client: Qdrant client or store
        embedder: Embedding model
        collection_name: Collection name (defaults to COLLECTION_NAME env var)
        corpus_docs: Optional corpus documents for BM25/hybrid retrieval
        scorer_plugins: Optional scorer plugins for hybrid retrieval
        
    Returns:
        Configured retriever instance
        
    Raises:
        ValueError: If retrieval strategy is unknown
    """
    # Get configuration from environment
    strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid").lower()
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rrf_k = int(os.getenv("RRF_K", "60"))
    bm25_variant = os.getenv("BM25_VARIANT", "okapi").lower()
    
    # Resolve collection name
    if collection_name is None:
        collection_name = os.getenv("COLLECTION_NAME", "Sentio_docs")
    
    logger.info("Creating retriever with strategy: %s", strategy)
    
    # Get vector name from environment with backward compatibility
    vector_name = os.getenv("TEXT_VECTOR_NAME", "text-dense")
    logger.debug("Using vector name: %s", vector_name)
    
    # Create dense retriever (used directly or as part of hybrid)
    # Check if DenseRetriever accepts vector_name parameter
    import inspect
    from src.core.retrievers.dense import DenseRetriever
    
    dense_retriever_sig = inspect.signature(DenseRetriever.__init__)
    
    # Create dense retriever with appropriate parameters
    if "vector_name" in dense_retriever_sig.parameters:
        # DenseRetriever supports vector_name parameter
        dense_retriever = DenseRetriever(
            client=client,
            embedder=embedder,
            collection_name=collection_name,
            vector_name=vector_name,
        )
        logger.info("Created dense retriever with vector_name parameter")
    else:
        # DenseRetriever doesn't support vector_name parameter
        dense_retriever = DenseRetriever(
            client=client,
            embedder=embedder,
            collection_name=collection_name,
        )
        logger.info("Created dense retriever without vector_name parameter")
    
    if strategy == "dense":
        logger.info("Using dense retrieval strategy")
        return dense_retriever
    
    elif strategy == "bm25":
        logger.info("Using BM25 retrieval strategy with variant: %s", bm25_variant)
        if not corpus_docs:
            logger.warning("No corpus documents provided for BM25 retrieval, using empty corpus")
            corpus_docs = []
            
        return BM25Retriever(
            documents=corpus_docs,
            variant=bm25_variant,
            cache_dir=os.getenv("SPARSE_CACHE_DIR", ".sparse_cache"),
        )
    
    elif strategy == "pyserini":
        logger.info("Using Pyserini BM25 retrieval strategy")
        try:
            return PyseriniBM25Retriever(
                index_dir=os.getenv("BM25_INDEX_DIR", "indexes/lucene-index"),
                k1=0.9,
                b=0.4,
            )
        except RuntimeError as e:
            logger.error("Failed to initialize PyseriniBM25Retriever: %s", e)
            logger.warning("Falling back to in-memory BM25")
            
            if not corpus_docs:
                logger.warning("No corpus documents provided for BM25 fallback, using empty corpus")
                corpus_docs = []
                
            return BM25Retriever(
                documents=corpus_docs,
                variant=bm25_variant,
                cache_dir=os.getenv("SPARSE_CACHE_DIR", ".sparse_cache"),
            )
    
    elif strategy == "hybrid":
        logger.info("Using hybrid retrieval strategy")
        
        # Try to use Pyserini if available
        sparse_retriever = None
        use_pyserini = os.path.isdir(os.getenv("BM25_INDEX_DIR", "indexes/lucene-index"))
        
        if use_pyserini:
            try:
                sparse_retriever = PyseriniBM25Retriever(
                    index_dir=os.getenv("BM25_INDEX_DIR", "indexes/lucene-index"),
                )
                logger.info("Using Pyserini for sparse retrieval in hybrid strategy")
            except RuntimeError as e:
                logger.error("Failed to initialize PyseriniBM25Retriever: %s", e)
                sparse_retriever = None
        
        return HybridRetriever(
            dense_retriever=dense_retriever,
            corpus_docs=corpus_docs,
            rrf_k=rrf_k,
            scorer_plugins=scorer_plugins,
            sparse_retriever=sparse_retriever,
        )
    
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")


def create_retriever_for_graph() -> BaseRetriever:
    """Create a retriever for the LangGraph pipeline.
    
    This is a convenience function that creates all necessary components
    and returns a configured retriever based on environment variables.
    
    Returns:
        Configured retriever instance
    """
    from src.core.embeddings import get_embedder
    from src.utils.settings import settings
    
    # Create embedder
    embedder = get_embedder(
        model_name=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3"),
        api_key=os.getenv("EMBEDDING_MODEL_API_KEY", ""),
    )
    
    # Create Qdrant client
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    # Create retriever
    return create_retriever_from_env(
        client=client,
        embedder=embedder,
        collection_name=os.getenv("COLLECTION_NAME", "Sentio_docs"),
    ) 