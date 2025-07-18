"""
State definition for RAG pipeline LangGraph.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from core.models.document import Document


class RAGState(BaseModel):
    """
    State representation for RAG pipeline in LangGraph.
    
    This model captures all state information needed throughout the RAG pipeline
    execution, from query normalization to final response generation.
    
    Attributes:
        query: Original user query string.
        normalized_query: Preprocessed/normalized version of the query.
        retrieved_documents: Documents retrieved from the vector store.
        reranked_documents: Documents after reranking.
        context: Selected context prepared for the LLM.
        response: Generated LLM response.
        metadata: Optional metadata for tracking, debugging and evaluation.
    """
    
    query: str = Field(..., description="Original user query")
    normalized_query: str = Field("", description="Preprocessed/normalized query")
    retrieved_documents: List[Document] = Field(
        default_factory=list, 
        description="Documents retrieved from vector store"
    )
    reranked_documents: List[Document] = Field(
        default_factory=list, 
        description="Documents after reranking step"
    )
    context: str = Field("", description="Selected context prepared for LLM")
    response: str = Field("", description="Generated response text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for tracking and evaluation"
    )
    
    class Config:
        arbitrary_types_allowed = True 