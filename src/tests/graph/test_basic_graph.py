"""
Tests for the basic RAG graph implementation.

These tests verify the functionality of the basic RAG graph pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.graph import build_basic_graph
from core.graph.state import RAGState
from core.models.document import Document


def test_rag_state_initialization():
    """Test RAGState initialization."""
    state = RAGState(query="test query")
    assert state.query == "test query"
    assert state.normalized_query == ""
    assert len(state.retrieved_documents) == 0
    assert len(state.reranked_documents) == 0
    assert state.context == ""
    assert state.response == ""


@patch("core.retrievers.get_retriever")
@patch("core.rerankers.get_reranker")
def test_basic_graph_creation(mock_get_reranker, mock_get_retriever):
    """Test basic graph creation."""
    # Setup mock retriever and reranker
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_get_retriever.return_value = mock_retriever
    mock_get_reranker.return_value = mock_reranker
    
    # Create graph
    graph = build_basic_graph()
    
    # Check graph structure (this is a basic smoke test)
    assert graph is not None


@patch("core.retrievers.get_retriever")
@patch("core.rerankers.get_reranker")
def test_graph_executes_all_nodes(mock_get_reranker, mock_get_retriever):
    """Test that the graph executes all nodes in the pipeline."""
    # Setup mock retriever
    test_docs = [
        Document(text="Doc 1", metadata={"source": "test1.txt"}),
        Document(text="Doc 2", metadata={"source": "test2.txt"}),
    ]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = test_docs
    
    # Setup mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = test_docs
    
    mock_get_retriever.return_value = mock_retriever
    mock_get_reranker.return_value = mock_reranker
    
    # Create graph
    graph = build_basic_graph(retriever=mock_retriever, reranker=mock_reranker)
    
    # Execute graph
    state = RAGState(query="test query")
    config = {"configurable": {"thread_id": "test_thread"}}
    result = graph.invoke(state, config)
    
    # Verify retriever was called
    mock_retriever.retrieve.assert_called_once()
    
    # Verify reranker was called
    mock_reranker.rerank.assert_called_once()
    
    # Check result state
    assert result.query == "test query"
    assert result.normalized_query == "test query"
    assert len(result.retrieved_documents) == 2
    assert len(result.reranked_documents) == 2
    assert result.context != ""
    assert result.response != ""
    
    # Should contain placeholders in the response
    assert "placeholder response" in result.response
    assert "test query" in result.response 