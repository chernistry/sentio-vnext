# Sentio vNext

LangGraph-based implementation of the Sentio RAG system.


## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [CLI Commands](#cli-commands)
- [Security Considerations](#security-considerations)
- [Planned Improvements](#planned-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Sentio vNext is a boilerplate of an Retrieval-Augmented Generation (RAG) system built with a LangGraph-inspired architecture. It provides high-performance vector retrieval, intelligent reranking, and streaming responses with an OpenAI-compatible API.

This system is designed for enterprise deployment with comprehensive monitoring, logging, and security considerations.

## Project Structure

```
src/
├── api/                # FastAPI entrypoints
├── core/
│   ├── chunking/       # Sentence/para splitters
│   ├── embeddings/     # Provider adapters
│   ├── ingest/         # Ingestion CLI + tasks
│   ├── llm/            # Chat/Prompt builders
│   ├── retrievers/     # Dense / hybrid search
│   ├── rerankers/      # Cross-encoders etc.
│   ├── vector_store/   # Vector store adapters
│   ├── models/         # Data models
│   └── plugins/        # Optional hooks
├── utils/              # Settings, logging helpers
├── tests/              # Pytest with pytest-asyncio
└── cli/                # Typer commands
```

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- Access to a Qdrant vector database instance
- API keys for selected embedding and LLM providers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentio-vnext

# Install dependencies
poetry install

# Set up pre-commit hooks
pre-commit install
```

## Configuration

Provide Qdrant credentials in environment variables:
```bash
export QDRANT_URL=https://your-qdrant-instance.cloud
export QDRANT_API_KEY=your-api-key
```

Additional configuration options are managed in `src/utils/settings.py` and can be set via environment variables:

### Vector Store
- `VECTOR_STORE`: Vector store backend. Default: `qdrant`.
- `COLLECTION_NAME`: Name for the vector collection. Default: `Sentio_docs`.

### Embeddings
- `EMBEDDER_NAME`: Embedding provider. Default: `jina`.
- `EMBEDDING_MODEL`: Specific embedding model. Default: `jina-embeddings-v3`.
- `EMBEDDING_MODEL_API_KEY`: API key for the embedding model.

### Reranker
- `RERANKER_MODEL`: Reranker model. Default: `jina-reranker-m0`.
- `RERANKER_URL`: Reranker API endpoint. Default: `https://api.jina.ai/v1/rerank`.
- `RERANKER_TIMEOUT`: Timeout for the reranker service. Default: `30`.

### Language Model (LLM)
- `LLM_PROVIDER`: LLM provider. Default: `openai`.
- `OPENAI_API_KEY`: API key for OpenAI.
- `OPENAI_MODEL`: OpenAI model name. Default: `gpt-3.5-turbo`.
- `CHAT_LLM_BASE_URL`: Base URL for the chat LLM. Default: `https://api.openai.com/v1`.
- `CHAT_LLM_MODEL`: Chat LLM model name. Default: `gpt-3.5-turbo`.
- `CHAT_LLM_API_KEY`: API key for the chat LLM.

### OpenRouter
- `OPENROUTER_REFERER`: Referer for OpenRouter.
- `OPENROUTER_TITLE`: Title for OpenRouter.

### Chunking
- `CHUNK_SIZE`: Chunk size in tokens. Default: `512`.
- `CHUNK_OVERLAP`: Chunk overlap in tokens. Default: `64`.
- `CHUNKING_STRATEGY`: Chunking strategy (`recursive`, `fixed`, etc.). Default: `recursive`.

### Retrieval
- `TOP_K_RETRIEVAL`: Number of documents to retrieve initially. Default: `10`.
- `TOP_K_RERANK`: Number of documents to rerank. Default: `5`.
- `MIN_RELEVANCE_SCORE`: Minimum relevance score for retrieval. Default: `0.05`.

## Usage

```bash
# Run the API server
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Ingest data
poetry run sentio ingest path/to/documents

# Run the UI
poetry run sentio ui

# Start the API server
poetry run sentio api

# Run a specific pipeline
poetry run sentio run pipeline_name

# Launch the studio
poetry run sentio studio
```

## API Endpoints

- `POST /chat`: Process a chat request using RAG. **(Note: Currently returns a stub response)**
- `POST /embed`: Embed a document and store it in the vector database.
- `GET /health`: Health check endpoint.
- `POST /clear`: Clear the vector store collection.
- `GET /info`: Get system information.

## CLI Commands

The `sentio` CLI provides several subcommands:
- `ingest`: Ingest documents into the vector store.
- `ui`: Launch the Streamlit UI.
- `api`: Start the FastAPI server.
- `run`: Run specific processing pipelines.
- `studio`: Launch the development studio.

## Security Considerations

- All API keys and sensitive configuration should be stored as environment variables.
- The application should be deployed behind appropriate authentication and authorization mechanisms.
- Network access should be restricted to authorized personnel only.
- Regular security audits should be conducted.
- Data encryption at rest and in transit is required for sensitive information.

## Planned Improvements

- Connect the `/chat` endpoint to a fully implemented LangGraph pipeline.
- Add support for additional vector stores (e.g., Milvus, Weaviate).
- Implement more sophisticated and adaptive chunking strategies.
- Add comprehensive evaluation and benchmarking tools for RAG performance.
- Introduce persistent job queues for large ingestion tasks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

All contributions must follow the corporate coding standards and be reviewed by at least one senior team member.

## License
Creative Commons Attribution-NonCommercial 4.0 International

Refer to the LICENSE file for the full license text.

© 2025 Alex Chernysh. All rights reserved.