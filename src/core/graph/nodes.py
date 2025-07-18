"""
Node implementations for RAG LangGraph pipeline.

This module contains the node functions that implement each step of the RAG pipeline.
Each node takes a RAGState and returns an updated RAGState.
"""

from typing import Dict, Any, List, TypeVar, cast
import logging
from copy import deepcopy

from core.models.document import Document
from core.retrievers.base import BaseRetriever
from core.rerankers.cross_encoder import CrossEncoderReranker
from core.graph.state import RAGState

logger = logging.getLogger(__name__)

T = TypeVar("T")


def normalize_query(state: RAGState) -> Dict[str, Any]:
    """
    Normalize and preprocess the query.
    
    Args:
        state: Current RAG pipeline state
        
    Returns:
        Dict with updated normalized_query
    """
    # Simple normalization for now - can be extended with stemming, stopword removal, etc.
    normalized = state.query.strip().lower()
    logger.debug("Normalized query: %s → %s", state.query, normalized)
    
    return {"normalized_query": normalized}


def retrieve_documents(
    state: RAGState, 
    retriever: BaseRetriever,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Retrieve relevant documents using the configured retriever.
    
    Args:
        state: Current RAG pipeline state
        retriever: Document retriever implementation
        top_k: Maximum number of documents to retrieve
        
    Returns:
        Dict with retrieved documents
    """
    query = state.normalized_query or state.query
    docs = retriever.retrieve(query, top_k=top_k)
    
    logger.info("Retrieved %d documents for query: %s", len(docs), query)
    
    return {"retrieved_documents": docs}


def rerank_documents(
    state: RAGState, 
    reranker: CrossEncoderReranker,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Rerank documents using the cross-encoder reranker.
    
    Args:
        state: Current RAG pipeline state
        reranker: Document reranker implementation
        top_k: Maximum number of documents to return after reranking
        
    Returns:
        Dict with reranked documents
    """
    if not state.retrieved_documents:
        logger.warning("No documents to rerank")
        return {"reranked_documents": []}
    
    query = state.normalized_query or state.query
    reranked_docs = reranker.rerank(
        query=query,
        docs=state.retrieved_documents,
        top_k=top_k
    )
    
    logger.info("Reranked %d documents", len(reranked_docs))
    
    return {"reranked_documents": reranked_docs}


def prepare_context(state: RAGState, max_tokens: int = 3000) -> Dict[str, Any]:
    """
    Prepare context from reranked documents for LLM.
    
    Args:
        state: Current RAG pipeline state
        max_tokens: Approximate maximum number of tokens to include
        
    Returns:
        Dict with prepared context string
    """
    docs = state.reranked_documents or state.retrieved_documents
    if not docs:
        logger.warning("No documents available for context preparation")
        return {"context": ""}
    
    # Simple context preparation - concatenate with document markers
    # This can be improved with more sophisticated context preparation strategies
    contexts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"document_{i}")
        score = doc.metadata.get("rerank_score", 0.0)
        contexts.append(f"[Document {i+1}] {doc.text}\nSource: {source} (Score: {score:.3f})")
        
    context = "\n\n".join(contexts)
    
    # Very crude token limitation - can be improved with actual tokenization
    if len(context) > max_tokens * 4:  # Rough approximation of token/char ratio
        context = context[:max_tokens * 4] + "..."
    
    return {"context": context}


def generate_response(state: RAGState) -> Dict[str, Any]:
    """
    Generate response based on context and query (placeholder).
    
    Args:
        state: Current RAG pipeline state
        
    Returns:
        Dict with generated response
    """
    # This is a placeholder for actual LLM integration
    # Will be replaced with real LLM call in Stage 9
    
    placeholder_response = (
        f"This is a placeholder response for query: '{state.query}'. "
        f"Context length: {len(state.context)} characters. "
        f"Number of documents: {len(state.reranked_documents or state.retrieved_documents)}"
    )
    
    logger.info("Generated placeholder response of %d chars", len(placeholder_response))
    
    return {"response": placeholder_response} 