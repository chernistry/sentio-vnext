from __future__ import annotations

"""Factory for building LangGraph RAG pipelines."""

import logging
import os
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.core.graph.state import RAGState
from src.core.graph.nodes import (
    create_retriever_node,
    create_reranker_node,
    create_document_selector_node,
    create_generator_node,
)
from src.core.retrievers.base import BaseRetriever
from src.core.rerankers.base import Reranker
from src.core.retrievers.factory import create_retriever_for_graph

logger = logging.getLogger(__name__)


class GraphConfig:
    """Configuration for building a LangGraph RAG pipeline.
    
    This class holds the configuration for building a LangGraph RAG pipeline,
    including the retriever, reranker, and LLM components.
    
    Attributes:
        retriever: The retriever component
        reranker: Optional reranker component
        llm: The language model for generation
        retrieval_top_k: Number of documents to retrieve
        reranking_top_k: Number of documents to return after reranking
        selection_top_k: Number of documents to select for generation
        max_tokens: Maximum tokens for selected documents
        prompt_template: Optional custom prompt template
    """
    
    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[BaseChatModel] = None,
        retrieval_top_k: Optional[int] = None,
        reranking_top_k: Optional[int] = None,
        selection_top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        prompt_template: Optional[ChatPromptTemplate] = None,
    ):
        """Initialize a GraphConfig instance.
        
        Args:
            retriever: The retriever component (if None, created from env)
            reranker: Optional reranker component (if None, created from env if USE_RERANKER=true)
            llm: The language model for generation
            retrieval_top_k: Number of documents to retrieve
            reranking_top_k: Number of documents to return after reranking
            selection_top_k: Number of documents to select for generation
            max_tokens: Maximum tokens for selected documents
            prompt_template: Optional custom prompt template
        """
        # Create retriever from environment if not provided
        self.retriever = retriever or create_retriever_for_graph()
        
        # Create reranker from environment if not provided and enabled
        self.reranker = reranker
        if self.reranker is None and os.getenv("USE_RERANKER", "true").lower() == "true":
            from src.core.rerankers import get_reranker
            try:
                self.reranker = get_reranker(kind="jina")
                logger.info("Created reranker from environment configuration")
            except Exception as e:
                logger.error("Failed to create reranker: %s", e)
                self.reranker = None
        
        # Set other parameters from environment if not provided
        self.llm = llm
        self.retrieval_top_k = retrieval_top_k or int(os.getenv("RETRIEVAL_TOP_K", "10"))
        self.reranking_top_k = reranking_top_k or int(os.getenv("RERANKING_TOP_K", "5"))
        self.selection_top_k = selection_top_k or int(os.getenv("SELECTION_TOP_K", "3"))
        self.max_tokens = max_tokens or 2000
        self.prompt_template = prompt_template


def build_basic_graph(config: Optional[Union[GraphConfig, Dict[str, Any]]] = None) -> StateGraph:
    """Build a basic LangGraph RAG pipeline.
    
    This function builds a basic LangGraph RAG pipeline with the following nodes:
    - retriever: Retrieves documents from the vector store
    - reranker: Reranks documents (optional)
    - selector: Selects documents for generation
    - generator: Generates a response
    
    Args:
        config: Configuration for the graph (if None, created from env)
               Can be either a GraphConfig object or a dict with configuration
        
    Returns:
        A compiled LangGraph StateGraph
    """
    # Handle different config types
    if config is None:
        # Create config from environment if not provided
        graph_config = GraphConfig()
    elif isinstance(config, dict):
        # Convert dict to GraphConfig
        logger.info("Converting dict config to GraphConfig")
        # Extract known parameters from dict
        graph_config = GraphConfig(
            retrieval_top_k=config.get("retrieval_top_k"),
            reranking_top_k=config.get("reranking_top_k"),
            selection_top_k=config.get("selection_top_k"),
            max_tokens=config.get("max_tokens"),
        )
    else:
        # Use provided GraphConfig
        graph_config = config
        
    # Create the graph
    graph = StateGraph(RAGState)
    
    # Create nodes
    retriever_node = create_retriever_node(
        retriever=graph_config.retriever,
        top_k=graph_config.retrieval_top_k,
    )
    
    selector_node = create_document_selector_node(
        top_k=graph_config.selection_top_k,
        max_tokens=graph_config.max_tokens,
    )
    
    # Add retriever node
    graph.add_node("retriever", retriever_node)
    
    # Conditionally add reranker node
    if graph_config.reranker:
        reranker_node = create_reranker_node(
            reranker=graph_config.reranker,
            top_k=graph_config.reranking_top_k,
        )
        graph.add_node("reranker", reranker_node)
        
        # Connect retriever to reranker
        graph.add_edge("retriever", "reranker")
        
        # Connect reranker to selector
        graph.add_node("selector", selector_node)
        graph.add_edge("reranker", "selector")
    else:
        # No reranker, connect retriever directly to selector
        graph.add_node("selector", selector_node)
        graph.add_edge("retriever", "selector")
    
    # Conditionally add generator node
    if graph_config.llm:
        generator_node = create_generator_node(
            llm=graph_config.llm,
            prompt_template=graph_config.prompt_template,
        )
        graph.add_node("generator", generator_node)
        graph.add_edge("selector", "generator")
        graph.add_edge("generator", END)
    else:
        # No generator, end after selector
        graph.add_edge("selector", END)
    
    # Set the entry point
    graph.set_entry_point("retriever")
    
    # Compile the graph
    return graph.compile()


def build_streaming_graph(config: Optional[Union[GraphConfig, Dict[str, Any]]] = None) -> StateGraph:
    """Build a streaming LangGraph RAG pipeline.
    
    This function builds a LangGraph RAG pipeline with streaming response
    support. This is a placeholder implementation that will be expanded
    in the future.
    
    Args:
        config: Configuration for the graph (if None, created from env)
               Can be either a GraphConfig object or a dict with configuration
        
    Returns:
        A compiled LangGraph StateGraph with streaming support
    """
    # For now, just use the basic graph
    # In a real implementation, this would configure streaming response
    return build_basic_graph(config) 