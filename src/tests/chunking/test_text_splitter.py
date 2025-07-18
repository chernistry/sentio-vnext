"""
Tests for the text_splitter module.
"""

import pytest
import re
import asyncio
import unicodedata

from src.core.chunking.text_splitter import (
    SentenceSplitter, 
    TOKEN_RE, 
    SemanticSplitter, 
    FixedSplitter, 
    ParagraphSplitter,
    TextChunker,
    ChunkingStrategy,
    ChunkingError
)
from src.core.models.document import Document


# ---- Base tests for SentenceSplitter ----

def test_token_regex():
    """Test the token regex pattern."""
    text = "Hello, world! This is a test."
    tokens = TOKEN_RE.findall(text)
    assert len(tokens) == 9
    assert tokens == ["Hello", ",", "world", "!", "This", "is", "a", "test", "."]


def test_sentence_splitter_init_validation():
    """Test validation in SentenceSplitter initialization."""
    # Valid parameters should not raise exceptions
    SentenceSplitter(chunk_size=100, chunk_overlap=50)
    
    # Invalid parameters should raise ValueError
    with pytest.raises(ValueError):
        SentenceSplitter(chunk_size=0)
        
    with pytest.raises(ValueError):
        SentenceSplitter(chunk_size=100, chunk_overlap=-1)
        
    with pytest.raises(ValueError):
        SentenceSplitter(chunk_size=100, chunk_overlap=100)
        
    with pytest.raises(ValueError):
        SentenceSplitter(chunk_size=100, chunk_overlap=150)


def test_split_empty_text():
    """Test splitting empty text."""
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []


def test_split_short_text():
    """Test splitting text shorter than chunk_size."""
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
    text = "This is a short text that should fit in one chunk."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_text():
    """Test splitting text longer than chunk_size."""
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    
    # Reconstruct original text (minus potential whitespace differences)
    reconstructed = " ".join(chunks)
    assert re.sub(r'\s+', ' ', reconstructed).strip() == re.sub(r'\s+', ' ', text).strip()


def test_very_long_sentence():
    """Test handling of a single sentence that exceeds chunk_size."""
    splitter = SentenceSplitter(chunk_size=10, chunk_overlap=2)
    long_sentence = "Thisisaverylongsentencewithnospacesthatshouldbesplitatthewordlevel."
    chunks = splitter.split_text(long_sentence)
    assert len(chunks) > 1
    
    # Check that we didn't lose any content
    combined = "".join(chunks).replace(" ", "")
    assert combined == long_sentence


def test_unicode_text():
    """Test splitting text with Unicode characters."""
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
    unicode_text = "こんにちは世界。这是一个测试。Привет, мир!"
    chunks = splitter.split_text(unicode_text)
    
    # Check that we didn't lose any content (normalize for comparison)
    combined = " ".join(chunks)
    assert unicodedata.normalize("NFC", combined) == unicodedata.normalize("NFC", unicode_text)


def test_code_snippets():
    """Test handling of code snippets with special characters."""
    splitter = SentenceSplitter(chunk_size=50, chunk_overlap=10)
    code = """def example_function():
    # This is a comment
    for i in range(10):
        print(f"Value: {i}")
    return True"""
    
    chunks = splitter.split_text(code)
    
    # Check that all content is preserved
    reconstructed = " ".join(chunks)
    assert re.sub(r'\s+', ' ', reconstructed).strip() == re.sub(r'\s+', ' ', code).strip()


def test_split_documents():
    """Test splitting a list of documents."""
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
    
    docs = [
        Document(text="This is document one. With two sentences.", metadata={"source": "doc1"}),
        Document(text="This is document two.", metadata={"source": "doc2"})
    ]
    
    chunks = splitter.split_documents(docs)
    
    # Verify chunks were created
    assert len(chunks) >= 2
    
    # Check that metadata was preserved and parent_id was set
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "parent_id" in chunk.metadata
        assert chunk.metadata["source"] in ["doc1", "doc2"]
        
    # Verify text content was preserved
    all_text = " ".join(chunk.text for chunk in chunks)
    assert "document one" in all_text
    assert "document two" in all_text 


# ---- Tests for other splitter strategies ----

def test_semantic_splitter():
    """Test the semantic splitter with markdown content."""
    text = """# Heading 1
    
Some paragraph text here. This should be kept together as a semantic unit.

## Heading 2

* List item 1
* List item 2

Another paragraph with some context."""

    splitter = SemanticSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    # Verify we got some chunks
    assert len(chunks) > 0
    
    # Check that each chunk is smaller than chunk_size
    for chunk in chunks:
        assert len(chunk) <= 100
        
    # Verify that all content is preserved
    all_text = " ".join(chunks)
    for phrase in ["Heading 1", "paragraph text", "Heading 2", "List item", "Another paragraph"]:
        assert phrase in all_text


def test_fixed_splitter():
    """Test the fixed splitter with a long text."""
    long_text = " ".join(["word" + str(i) for i in range(100)])
    
    splitter = FixedSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(long_text)
    
    # Verify we got multiple chunks
    assert len(chunks) > 1
    
    # Check overlap
    words1 = set(chunks[0].split())
    words2 = set(chunks[1].split())
    intersection = words1.intersection(words2)
    
    # Should have some overlapping words
    assert len(intersection) > 0
    
    # All chunks should be <= chunk_size in length
    for chunk in chunks:
        assert len(chunk) <= 50


def test_paragraph_splitter():
    """Test paragraph splitter with multi-paragraph content."""
    text = """Paragraph one with some content.

Paragraph two with different content.

Paragraph three is here as well.

And here's the fourth paragraph."""

    splitter = ParagraphSplitter(chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    # Should produce at least 2 chunks given the size
    assert len(chunks) >= 2
    
    # Check content preservation
    combined = " ".join(chunks)
    for phrase in ["Paragraph one", "Paragraph two", "Paragraph three", "fourth paragraph"]:
        assert phrase in combined


# ---- Tests for TextChunker facade ----

@pytest.mark.asyncio
async def test_text_chunker_create():
    """Test the creation of TextChunker via factory method."""
    chunker = await TextChunker.create(
        chunk_size=100,
        chunk_overlap=20,
        strategy=ChunkingStrategy.HYBRID,
        min_chunk_size=10,
        preserve_code_blocks=True
    )
    
    assert chunker is not None
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 20
    assert chunker.strategy == ChunkingStrategy.HYBRID
    assert chunker.min_chunk_size == 10
    assert chunker.preserve_code_blocks is True
    
    # Check that stats are initialized
    assert chunker.stats["documents_processed"] == 0
    assert chunker.stats["total_chunks"] == 0


@pytest.mark.asyncio
async def test_chunker_split_with_strategy_selection():
    """Test that TextChunker selects appropriate strategies based on content."""
    chunker = await TextChunker.create(
        strategy=ChunkingStrategy.HYBRID,
        chunk_size=100,
        chunk_overlap=20
    )
    
    docs = [
        # Document with headings - should use SEMANTIC strategy
        Document(text="# Heading\n\nParagraph content.\n\n## Subheading\n\nMore content.",
                 metadata={"type": "markdown"}),
        
        # Document with short sentences - should use SENTENCE strategy
        Document(text="Short sentence one. Short sentence two. Short sentence three.",
                 metadata={"type": "plain"})
    ]
    
    result = chunker.split(docs)
    
    # Verify documents were processed
    assert chunker.stats["documents_processed"] == 2
    assert len(result) > 0
    
    # Verify strategy usage stats were updated
    assert sum(chunker.stats["strategy_usage"].values()) > 0


def test_chunker_special_content_handling():
    """Test that TextChunker properly handles special content like code blocks."""
    text_with_code = """
    # Example Document
    
    Here's some regular text.
    
    ```python
    def test_function():
        print("Hello world!")
        return True
    ```
    
    More text after the code block.
    """
    
    doc = Document(text=text_with_code, metadata={"type": "markdown"})
    
    chunker = TextChunker(
        splitters={
            ChunkingStrategy.SENTENCE: SentenceSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.SEMANTIC: SemanticSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.FIXED: FixedSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.PARAGRAPH: ParagraphSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.HYBRID: SentenceSplitter(chunk_size=100, chunk_overlap=20),
        },
        chunk_size=100,
        chunk_overlap=20,
        strategy=ChunkingStrategy.SEMANTIC,
        min_chunk_size=10,
        max_chunk_size=200,
        preserve_code_blocks=True,
        preserve_tables=True
    )
    
    # Preprocess should identify code blocks
    processed_text, regions = chunker._preprocess_text(text_with_code)
    assert len(regions["code_blocks"]) > 0


def test_chunker_stats_tracking():
    """Test that TextChunker properly tracks and reports statistics."""
    chunker = TextChunker(
        splitters={
            ChunkingStrategy.SENTENCE: SentenceSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.SEMANTIC: SemanticSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.FIXED: FixedSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.PARAGRAPH: ParagraphSplitter(chunk_size=100, chunk_overlap=20),
            ChunkingStrategy.HYBRID: SentenceSplitter(chunk_size=100, chunk_overlap=20),
        },
        chunk_size=100,
        chunk_overlap=20,
        strategy=ChunkingStrategy.SENTENCE,
        min_chunk_size=10,
        max_chunk_size=200,
        preserve_code_blocks=True,
        preserve_tables=True
    )
    
    doc = Document(text="Test sentence one. Test sentence two.", metadata={"source": "test"})
    
    result = chunker.split([doc])
    
    # Check stats were updated
    assert chunker.stats["documents_processed"] == 1
    assert chunker.stats["total_chunks"] > 0
    assert chunker.stats["strategy_usage"][ChunkingStrategy.SENTENCE.value] > 0
    
    # Test stats reset
    chunker.reset_stats()
    assert chunker.stats["documents_processed"] == 0
    assert chunker.stats["total_chunks"] == 0
    
    # Test detailed stats
    detailed_stats = chunker.get_stats()
    assert "config" in detailed_stats
    assert detailed_stats["config"]["chunk_size"] == 100 