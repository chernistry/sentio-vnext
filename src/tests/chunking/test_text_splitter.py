"""
Tests for the text_splitter module.
"""

import pytest
import re
import unicodedata

from src.core.chunking.text_splitter import SentenceSplitter, TOKEN_RE
from src.core.models.document import Document


def test_token_regex():
    """Test the token regex pattern."""
    text = "Hello, world! This is a test."
    tokens = TOKEN_RE.findall(text)
    assert len(tokens) == 8
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