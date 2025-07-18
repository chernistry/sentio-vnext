#!/usr/bin/env python
"""
Script to ingest documents into the vector database.

This script provides a command-line interface to the document ingestion
functionality, allowing users to easily ingest documents from a directory.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from core.ingest import ingest_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point with argument parsing and configuration."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector database for RAG",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("SAMPLE_DOCS_FOLDER", "data/raw"),
        help="Directory containing documents to ingest (default: data/raw or SAMPLE_DOCS_FOLDER env var)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="Sentio_docs",
        help="Vector store collection name (default: Sentio_docs)"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Text chunk size in tokens (default: 512)"
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=64,
        help="Overlap between chunks in tokens (default: 64)"
    )
    
    parser.add_argument(
        "--chunking_strategy",
        type=str,
        choices=["recursive", "fixed"],
        default="recursive",
        help="Chunking strategy to use (default: recursive)"
    )

    parser.add_argument(
        "--embedder",
        type=str,
        default="jina",
        help="Embedding provider to use (default: jina)"
    )

    parser.add_argument(
        "--vector_store",
        type=str,
        default="qdrant",
        help="Vector store backend to use (default: qdrant)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Convert data_dir to absolute path if relative
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    
    logger.info(f"Starting document ingestion from {data_dir}")
    
    try:
        stats = await ingest_directory(
            data_dir=data_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunking_strategy=args.chunking_strategy,
            embedder_name=args.embedder,
            vector_store_name=args.vector_store,
        )
        
        logger.info("🎉 Ingestion completed successfully!")
        logger.info(f"   Documents processed: {stats['documents_processed']}")
        logger.info(f"   Chunks created: {stats['chunks_created']}")
        logger.info(f"   Embeddings generated: {stats['embeddings_generated']}")
        logger.info(f"   Bytes processed: {stats['bytes_processed']}")
        
    except KeyboardInterrupt:
        logger.info("Ingestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 