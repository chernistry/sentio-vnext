#!/usr/bin/env python3
"""
Test script for demonstrating the chunking functionality on actual documents.
"""

import sys
from pathlib import Path
import time

# Add project src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core.chunking.text_splitter import SentenceSplitter
from src.core.models.document import Document


def read_text_file(file_path: str) -> str:
    """Read text file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    """Main function to demonstrate chunking functionality."""
    # Path to test document
    test_file_path = "../legacy/root/data/raw/python_for_osint.txt"
    
    try:
        # Read document
        print(f"Reading document: {test_file_path}")
        text = read_text_file(test_file_path)
        print(f"Document size: {len(text)} characters")
        
        # Create document object
        doc = Document(
            text=text,
            metadata={"source": test_file_path, "type": "text/plain"}
        )
        print(f"Created document with ID: {doc.id}")
        
        # Initialize sentence splitter
        print("\nInitializing SentenceSplitter...")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        
        # Time the chunking operation
        start_time = time.time()
        print("Splitting document into chunks...")
        
        # Create chunks
        chunks = splitter.split_documents([doc])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        print(f"\nChunking completed in {duration:.2f} seconds")
        print(f"Created {len(chunks)} chunks from document")
        
        # Show some sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}/{len(chunks)}")
            print(f"Chunk ID: {chunk.id}")
            print(f"Parent ID: {chunk.metadata.get('parent_id')}")
            print(f"Source: {chunk.metadata.get('source')}")
            print("Text preview (first 100 chars):", chunk.text[:100] + "...")
            print(f"Chunk length: {len(chunk.text)} chars")
            
        # Calculate average chunk size
        avg_size = sum(len(chunk.text) for chunk in chunks) / len(chunks)
        print(f"\nAverage chunk size: {avg_size:.2f} characters")
        
        # Verify content integrity
        original_text = doc.text.replace(" ", "").replace("\n", "")
        chunked_text = "".join(chunk.text for chunk in chunks).replace(" ", "").replace("\n", "")
        
        content_match = original_text == chunked_text
        print(f"Content integrity check: {'PASSED' if content_match else 'FAILED'}")
        
        if not content_match:
            # Check what percentage of content is preserved
            orig_len = len(original_text)
            chunk_len = len(chunked_text)
            preserved = min(orig_len, chunk_len) / max(orig_len, chunk_len) * 100
            print(f"Content preservation: {preserved:.2f}%")
            print(f"Original length: {orig_len}, Chunked length: {chunk_len}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 