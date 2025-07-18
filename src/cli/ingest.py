"""
Ingestion CLI for Sentio vNext.

This module provides the command-line interface for ingesting documents into 
the Sentio system. It reads files from a specified path, splits them using the
TextChunker, and stores the chunks on disk for later processing.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

import asyncio
import typer
from typing_extensions import Annotated
import pandas as pd

from src.core.chunking import TextChunker, ChunkingStrategy
from src.core.models.document import Document
from src.utils.settings import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create Typer CLI app
ingest_app = typer.Typer(help="Document ingestion commands")

# Default data directory for storing chunks
DEFAULT_DATA_DIR = Path("data")


async def process_file(
    file_path: Path, 
    chunker: TextChunker,
    metadata: Optional[dict] = None
) -> List[Document]:
    """
    Process a single file by reading its content and chunking it.
    
    Args:
        file_path: Path to the file to process
        chunker: TextChunker instance to use for splitting
        metadata: Optional additional metadata to include
        
    Returns:
        List of Document objects containing the chunks
    """
    if metadata is None:
        metadata = {}
        
    try:
        # Read file content
        logger.info(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Create base document
        doc_metadata = {
            "source": str(file_path.resolve()),
            "filename": file_path.name,
            "file_ext": file_path.suffix.lower(),
            **metadata
        }
        
        base_doc = Document(text=content, metadata=doc_metadata)
        
        # Split document into chunks
        chunks = chunker.split([base_doc])
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        return chunks
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []


async def process_directory(
    directory_path: Path,
    chunker: TextChunker,
    recursive: bool = True,
    extensions: List[str] = [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"]
) -> List[Document]:
    """
    Process all files in a directory.
    
    Args:
        directory_path: Path to the directory to process
        chunker: TextChunker instance to use for splitting
        recursive: Whether to process subdirectories recursively
        extensions: List of file extensions to process
        
    Returns:
        List of Document objects containing the chunks from all processed files
    """
    all_chunks = []
    
    # Get all files in the directory
    if recursive:
        files = [p for p in directory_path.glob("**/*") if p.is_file() and p.suffix.lower() in extensions]
    else:
        files = [p for p in directory_path.glob("*") if p.is_file() and p.suffix.lower() in extensions]
        
    logger.info(f"Found {len(files)} files to process in {directory_path}")
    
    # Process each file
    for file_path in files:
        chunks = await process_file(file_path, chunker)
        all_chunks.extend(chunks)
        
    return all_chunks


def save_chunks_parquet(chunks: List[Document], output_path: Path) -> Path:
    """
    Save chunks to a parquet file for reproducible testing.
    
    Args:
        chunks: List of Document objects to save
        output_path: Directory to save the parquet file in
        
    Returns:
        Path to the created parquet file
    """
    # Convert chunks to a pandas DataFrame
    data = []
    for chunk in chunks:
        data.append({
            "id": chunk.id,
            "text": chunk.text,
            "source": chunk.metadata.get("source", ""),
            "filename": chunk.metadata.get("filename", ""),
            "parent_id": chunk.metadata.get("parent_id", ""),
            "chunk_index": chunk.metadata.get("chunk_index", 0),
        })
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save to parquet
    parquet_path = output_path / "chunks.parquet"
    df.to_parquet(parquet_path, index=False)
    
    return parquet_path


def save_chunks_disk(chunks: List[Document], output_path: Path) -> int:
    """
    Save chunks to disk as individual files.
    
    Args:
        chunks: List of Document objects to save
        output_path: Directory to save the chunk files in
        
    Returns:
        Number of chunks saved
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save each chunk as a separate file
    count = 0
    for chunk in chunks:
        # Create a filename based on the chunk ID
        chunk_file = output_path / f"chunk_{chunk.id}.txt"
        
        # Write the chunk text to the file
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk.text)
            
        # Write metadata as a separate JSON file
        metadata_file = output_path / f"chunk_{chunk.id}.meta.json"
        pd.Series(chunk.metadata).to_json(metadata_file)
        
        count += 1
        
    return count


@ingest_app.command("ingest")
def ingest_command(
    path: Annotated[Path, typer.Argument(help="Path to file or directory to ingest")],
    output_dir: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory for chunks")] = None,
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Process directories recursively")] = True,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Size of text chunks")] = None,
    chunk_overlap: Annotated[int, typer.Option("--overlap", help="Overlap between chunks")] = None,
    strategy: Annotated[str, typer.Option("--strategy", help="Chunking strategy to use")] = None,
):
    """
    Ingest documents into the system.
    
    This command reads files from the specified path, splits them into chunks using the
    configured TextChunker, and saves the chunks to disk for later processing.
    """
    if not path.exists():
        typer.echo(f"Error: Path '{path}' does not exist", err=True)
        raise typer.Exit(code=1)
    
    # Determine output directory
    output_path = output_dir or DEFAULT_DATA_DIR
    
    # Configure chunker based on settings or CLI arguments
    chunking_kwargs = settings.chunking_kwargs()
    
    # Override settings with CLI arguments if provided
    if chunk_size is not None:
        chunking_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        chunking_kwargs["chunk_overlap"] = chunk_overlap
    if strategy is not None:
        chunking_kwargs["strategy"] = strategy
        
    typer.echo(f"Initializing TextChunker with: {chunking_kwargs}")
    
    # Run in asyncio event loop
    async def run_ingest():
        # Create chunker
        chunker = await TextChunker.create(**chunking_kwargs)
        
        # Process input path
        chunks = []
        if path.is_file():
            chunks = await process_file(path, chunker)
        elif path.is_dir():
            chunks = await process_directory(path, chunker, recursive=recursive)
        else:
            typer.echo(f"Error: Path '{path}' is neither a file nor a directory", err=True)
            raise typer.Exit(code=1)
            
        if not chunks:
            typer.echo("No documents were successfully processed", err=True)
            raise typer.Exit(code=1)
            
        typer.echo(f"Successfully processed {len(chunks)} chunks")
        
        # Save chunks to disk
        typer.echo(f"Saving chunks to {output_path}")
        save_chunks_disk(chunks, output_path)
        
        # Save chunks to parquet for testing
        parquet_path = save_chunks_parquet(chunks, output_path)
        typer.echo(f"Saved chunks to {parquet_path} for deterministic testing")
        
        return chunks
    
    # Run the async function
    chunks = asyncio.run(run_ingest())
    typer.echo(f"Ingestion completed with {len(chunks)} chunks")


if __name__ == "__main__":
    ingest_app() 