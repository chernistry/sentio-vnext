"""
Factory module for creating and configuring LangGraph instances.

This module provides functions to create and configure different types of RAG graphs
using the LangGraph library. It serves as the main entry point for graph creation.
"""

import logging
from typing import Any, Dict, Optional, cast

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from core.retrievers.base import BaseRetriever
from core.retrievers import get_retriever
from core.rerankers.cross_encoder import CrossEncoderReranker
from core.rerankers import get_reranker
from core.graph.nodes import (
    normalize_query,
    retrieve_documents,
    rerank_documents,
    prepare_context,
    generate_response,
)
from core.graph.state import RAGState

logger = logging.getLogger(__name__)


def build_basic_graph(
    retriever: Optional[BaseRetriever] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    config: Optional[Dict[str, Any]] = None,
) -> StateGraph:
    """
    Build a basic RAG graph with the standard pipeline nodes.
    
    This creates a linear RAG pipeline with these steps:
    normalize → retrieve → rerank → prepare_context → generate
    
    Args:
        retriever: Optional retriever instance (will be created if not provided)
        reranker: Optional reranker instance (will be created if not provided)
        config: Optional configuration dictionary with parameters
        
    Returns:
        Compiled LangGraph StateGraph
    """
    config = config or {}
    retriever_cfg = config.get("retriever", {})
    reranker_cfg = config.get("reranker", {})
    
    # Create components if not provided
    if retriever is None:
        retriever = get_retriever(
            kind=retriever_cfg.get("kind", "hybrid"),
            top_k=retriever_cfg.get("top_k", 10),
        )
    
    if reranker is None:
        reranker = get_reranker(
            kind=reranker_cfg.get("kind", "cross-encoder"),
            model_name=reranker_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )
    
    # Create graph
    builder = StateGraph(RAGState)
    
    # Define nodes with partial functions to include dependencies
    builder.add_node("normalize", lambda state: normalize_query(state))
    builder.add_node(
        "retrieve", 
        lambda state: retrieve_documents(
            state, 
            retriever=retriever,
            top_k=retriever_cfg.get("top_k", 10),
        )
    )
    builder.add_node(
        "rerank", 
        lambda state: rerank_documents(
            state, 
            reranker=reranker,
            top_k=reranker_cfg.get("top_k", 5),
        )
    )
    builder.add_node(
        "prepare_context",
        lambda state: prepare_context(
            state,
            max_tokens=config.get("max_context_tokens", 3000),
        )
    )
    builder.add_node("generate", lambda state: generate_response(state))
    
    # Set the edges: linear pipeline
    builder.set_entry_point("normalize")
    builder.add_edge("normalize", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "prepare_context")
    builder.add_edge("prepare_context", "generate")
    builder.add_edge("generate", END)
    
    # Compile the graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    logger.info("Built basic RAG graph with %d nodes", len(builder.nodes))
    
    return graph 