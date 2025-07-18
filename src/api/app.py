"""
Sentio RAG System - FastAPI Application (LangGraph Version)

A modern RAG system built with LangGraph architecture:
- High-performance vector retrieval
- Intelligent reranking
- Streaming responses
- OpenAI-compatible API
- Comprehensive monitoring and logging
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.core.embeddings import get_embedder
from src.core.vector_store import get_vector_store
from src.core.ingest import DocumentIngestor
from src.utils.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Data Models
class ChatRequest(BaseModel):
    """
    Chat request with comprehensive validation.

    Attributes:
        question (str): User's question.
        history (Optional[List[Dict[str, str]]]): Chat history.
        top_k (Optional[int]): Number of results to return.
        temperature (Optional[float]): Generation temperature.
    """
    question: str = Field(
        ..., min_length=1, max_length=2000, description="User's question"
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=[], description="Chat history"
    )
    top_k: Optional[int] = Field(
        default=3, ge=1, le=20, description="Number of results to return"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Generation temperature"
    )

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class Source(BaseModel):
    """
    Source document with metadata.

    Attributes:
        text (str): Source text content.
        source (str): Document source identifier.
        score (float): Relevance score.
        metadata (Optional[Dict]): Additional metadata.
    """
    text: str = Field(..., description="Source text content")
    source: str = Field(..., description="Document source identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class ChatResponse(BaseModel):
    """
    Chat response with sources and metadata.

    Attributes:
        answer (str): Generated answer.
        sources (List[Source]): Source documents used.
        metadata (Optional[Dict]): Response metadata.
    """
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    metadata: Optional[Dict] = Field(default=None, description="Response metadata")


class EmbedRequest(BaseModel):
    """
    Request for document embedding/ingestion.

    Attributes:
        id (Optional[Union[int, str]]): Document identifier.
        content (str): Raw text content to embed.
        metadata (Optional[Dict]): Arbitrary metadata for the document.
    """
    id: Optional[Union[int, str]] = Field(
        default=None, description="Document identifier (auto-generated if omitted)"
    )
    content: str = Field(
        ..., min_length=1, max_length=50000, description="Raw text content to embed"
    )
    metadata: Optional[Dict] = Field(default=None, description="Arbitrary metadata for the document")


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status (str): Health status.
        timestamp (float): Timestamp of the health check.
        version (str): Application version.
        services (Dict[str, str]): Status of dependent services.
    """
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]


# Initialize FastAPI app
app = FastAPI(
    title="Sentio RAG API",
    description="LangGraph-based Retrieval-Augmented Generation system",
    version="3.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables
_ingestor: Optional[DocumentIngestor] = None
_vector_store: Optional[Any] = None


async def get_ingestor() -> DocumentIngestor:
    """Get or initialize the document ingestor."""
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor(
            collection_name=settings.collection_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunking_strategy=settings.chunking_strategy,
            embedder_name=settings.embedder_name,
            vector_store_name=settings.vector_store_name,
        )
        await _ingestor.initialize()
    return _ingestor


async def get_vector_store():
    """Get or initialize the vector store client."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store(
            name=settings.vector_store_name,
            collection_name=settings.collection_name,
            vector_size=get_embedder(settings.embedder_name).dimension,
        )
    return _vector_store


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for system status monitoring.
    """
    services = {}
    
    # Check vector store
    try:
        vector_store = await get_vector_store()
        if hasattr(vector_store, "health_check"):
            services["vector_store"] = "healthy" if vector_store.health_check() else "unhealthy"
        else:
            services["vector_store"] = "unknown"
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        services["vector_store"] = "unhealthy"
    
    # Check embedder
    try:
        embedder = get_embedder(settings.embedder_name)
        if embedder:
            services["embedder"] = "healthy"
        else:
            services["embedder"] = "unavailable"
    except Exception as e:
        logger.error(f"Embedder health check failed: {e}")
        services["embedder"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=time.time(),
        version="3.0.0",
        services=services
    )


@app.post("/embed")
async def embed_document(request: EmbedRequest) -> Dict[str, Any]:
    """
    Embed a document and store it in the vector database.
    """
    try:
        ingestor = await get_ingestor()
        
        doc_id = request.id or str(uuid.uuid4())
        
        # Create document with metadata
        from src.core.models.document import Document
        doc = Document(
            id=str(doc_id),
            text=request.content,
            metadata=request.metadata or {
                "source": "api_upload",
                "timestamp": time.time()
            }
        )
        
        # Process the single document (this will chunk, embed, and store)
        chunks = ingestor.chunker.split([doc])
        
        # Generate embeddings for chunks
        doc_embeddings = await ingestor._generate_embeddings(chunks)
        
        # Store chunks with embeddings
        await ingestor._store_chunks_with_embeddings(chunks, doc_embeddings)
        
        return {
            "status": "success",
            "id": doc_id,
            "chunks_created": len(chunks),
            "embeddings_generated": len(doc_embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error embedding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request using RAG.
    """
    try:
        # This is just a stub implementation that needs to be connected to the LangGraph pipeline
        # In a real implementation, this would call the query flow
        
        return ChatResponse(
            answer="This endpoint is not yet fully implemented in the LangGraph version.",
            sources=[],
            metadata={"status": "not_implemented"}
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_collection() -> Dict[str, str]:
    """
    Clear the vector store collection.
    """
    try:
        vector_store = await get_vector_store()
        
        # Method depends on the specific vector store implementation
        if hasattr(vector_store, "delete_collection"):
            # QdrantStore has delete_collection
            collection_name = settings.collection_name
            vector_store.delete_collection(collection_name)
            
            # Recreate empty collection
            vector_embedder = get_embedder(settings.embedder_name)
            vector_size = vector_embedder.dimension
            
            # Recreate collection
            if hasattr(vector_store, "create_collection"):
                vector_store.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size
                )
        
        return {"status": "success", "message": "Collection cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def system_info() -> Dict[str, Any]:
    """
    Get system information.
    """
    return {
        "name": "Sentio LangGraph RAG System",
        "version": "3.0.0",
        "configuration": {
            "collection_name": settings.collection_name,
            "embedding_provider": settings.embedder_name,
            "vector_store": settings.vector_store_name,
            "chunking_strategy": settings.chunking_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
    } 