"""
Factory module for creating and configuring LangGraph instances.

This module provides functions to create and configure different types of RAG graphs
using the LangGraph library. It serves as the main entry point for graph creation.
"""

import logging
from typing import Any, Dict, Optional, cast

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import RunnableConfig
# Опциональный импорт SqliteSaver
try:
    from langgraph_checkpoint_sqlite import SqliteSaver
    SQLITE_SAVER_AVAILABLE = True
except ImportError:
    SQLITE_SAVER_AVAILABLE = False
    print("Warning: langgraph_checkpoint_sqlite not found. Using in-memory state without persistence.")

from src.core.retrievers.base import BaseRetriever
from src.core.retrievers import get_retriever
from src.core.rerankers.cross_encoder import CrossEncoderReranker
from src.core.rerankers import get_reranker
from src.core.graph.nodes import (
    normalize_query,
    retrieve_documents,
    rerank_documents,
    prepare_context,
    generate_response,
)
from src.core.graph.state import RAGState

logger = logging.getLogger(__name__)


def build_basic_graph(config: RunnableConfig) -> StateGraph:
    """
    Build a basic RAG graph with the standard pipeline nodes.
    
    This creates a linear RAG pipeline with these steps:
    normalize → retrieve → rerank → prepare_context → generate
    
    Args:
        config: Configuration dictionary with parameters for the graph components
        
    Returns:
        Compiled LangGraph StateGraph
    """
    # Extract configuration
    retriever_cfg = config.get("retriever", {})
    reranker_cfg = config.get("reranker", {})
    
    # Create components based on configuration
    retriever = None
    reranker = None
    
    if 'mock' not in config.get('mode', ''):
        # Create retriever
        retriever = get_retriever(
            kind=retriever_cfg.get("kind", "dense"),
            **retriever_cfg,
        )
        
        # Create reranker
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
    
    # Компилируем граф
    if SQLITE_SAVER_AVAILABLE:
        memory = SqliteSaver.from_conn_string(":memory:")
        graph = builder.compile(checkpointer=memory)
    else:
        # Компиляция без чекпоинтера (хранение состояния только в памяти)
        graph = builder.compile()
    
    logger.info("Built basic RAG graph with %d nodes", len(builder.nodes))
    
    # Обертка для сохранения интерфейса
    original_invoke = graph.invoke
    
    def wrapped_invoke(state, *args, **kwargs):
        result = original_invoke(state, *args, **kwargs)
        
        # Если результат является словарем, преобразуем его в RAGState
        if isinstance(result, dict):
            # Создаем новый объект RAGState с данными из state
            new_state = RAGState(query=state.query)
            # Обновляем его полями из результата
            for key, value in result.items():
                setattr(new_state, key, value)
            return new_state
        return result
    
    graph.invoke = wrapped_invoke
    
    return graph


def build_streaming_graph(config: RunnableConfig) -> StateGraph:
    """
    Build a streaming RAG graph that supports streaming responses.
    
    Similar to the basic graph but optimized for streaming output.
    
    Args:
        config: Configuration dictionary with parameters for the graph components
        
    Returns:
        Compiled LangGraph StateGraph with streaming support
    """
    # Extract configuration
    retriever_cfg = config.get("retriever", {})
    reranker_cfg = config.get("reranker", {})
    
    # Create components based on configuration
    retriever = None
    reranker = None
    
    if 'mock' not in config.get('mode', ''):
        # Create retriever
        retriever = get_retriever(
            kind=retriever_cfg.get("kind", "dense"),
            **retriever_cfg,
        )
        
        # Create reranker
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
    builder.add_node("generate", lambda state: generate_response(state, stream=True))
    
    # Set the edges: linear pipeline
    builder.set_entry_point("normalize")
    builder.add_edge("normalize", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "prepare_context")
    builder.add_edge("prepare_context", "generate")
    builder.add_edge("generate", END)
    
    # Компилируем граф с поддержкой стриминга
    if SQLITE_SAVER_AVAILABLE:
        memory = SqliteSaver.from_conn_string(":memory:")
        graph = builder.compile(checkpointer=memory)
    else:
        graph = builder.compile()
    
    logger.info("Built streaming RAG graph with %d nodes", len(builder.nodes))
    
    return graph 