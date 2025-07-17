"""
Advanced text chunking with multiple strategies and optimization.

This module provides tools for splitting documents into smaller chunks
with configurable chunk size and overlap.
"""

import logging
import re
import asyncio

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Pattern

from collections import deque
from functools import lru_cache

from src.core.models.document import Document

# Pre‑compiled token regex for fast reuse
TOKEN_RE: Pattern[str] = re.compile(r'\b\w+\b|[^\w\s]')

logger = logging.getLogger(__name__)

# ==== DEFINITIONS & CONSTANTS ==== #
# --► ENUMS & EXCEPTIONS

class ChunkingStrategy(Enum):
    """Available chunking strategies for text chunking."""

    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    HYBRID = "hybrid"


class ChunkingError(Exception):
    """Exception raised for errors during text chunking."""


# ==== CORE PROCESSING MODULE ==== #
# --► DATA EXTRACTION & TRANSFORMATION

class BaseTextSplitter(ABC):
    """Abstract base class for all text splitter implementations."""
    __slots__: Tuple = ()

    @abstractmethod
    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The input text to split.
            metadata: Optional metadata for context.
            
        Returns:
            A list of text chunks.
        """
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: A list of Document objects to split.
            
        Returns:
            A list of Document objects representing the chunks.
        """
        splits = []
        for doc in documents:
            for chunk in self.split_text(doc.text):
                if not chunk.strip():  # Skip empty chunks
                    continue
                    
                # Create new metadata dict with original doc id
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["parent_id"] = doc.id
                
                # Create new document with chunk text and metadata
                splits.append(Document(
                    text=chunk,
                    metadata=chunk_metadata
                ))
        return splits


class SentenceSplitter(BaseTextSplitter):
    """Sentence-based chunking with intelligent boundary detection."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap")

    def __init__(
        self, 
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        """
        Initialize sentence splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    async def create(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> "SentenceSplitter":
        """
        Asynchronously create a SentenceSplitter instance.

        Args:
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Overlap tokens between chunks.

        Returns:
            A new instance of SentenceSplitter.
        """
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _smart_tokenizer(text: str) -> List[str]:
        """
        Tokenize text preserving semantic units (cached).

        Args:
            text: Raw input text.

        Returns:
            A list of tokens including punctuation.
        """
        return TOKEN_RE.findall(text)
        
    def _count_tokens(self, text: str) -> int:
        """
        Count the approximate number of tokens in text.
        
        Args:
            text: The text to count tokens in.
            
        Returns:
            Approximate token count.
        """
        return len(self._smart_tokenizer(text))

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Split text into chunks by sentence boundaries.
        
        Args:
            text: The input text to split.
            metadata: Optional metadata (not used in splitting but for compatibility)
            
        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            return []
            
        # Split text into sentences
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create chunks by grouping sentences
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self._count_tokens(sentence)
            
            # If a single sentence is larger than chunk_size, we need to split it
            if sentence_size > self.chunk_size:
                # If we have content in current_chunk, add it to chunks first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the long sentence into smaller pieces
                words = sentence.split()
                current_piece = []
                current_piece_size = 0
                
                for word in words:
                    word_size = self._count_tokens(word)
                    if current_piece_size + word_size > self.chunk_size:
                        if current_piece:
                            chunks.append(" ".join(current_piece))
                        current_piece = [word]
                        current_piece_size = word_size
                    else:
                        current_piece.append(word)
                        current_piece_size += word_size
                
                # Add the last piece if it exists
                if current_piece:
                    chunks.append(" ".join(current_piece))
                continue
            
            # Check if adding this sentence would exceed chunk_size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Handle overlap for next chunk
                # Find sentences to keep for overlap
                overlap_size = 0
                overlap_sentences = []
                
                for i in range(len(current_chunk) - 1, -1, -1):
                    sentence_to_keep = current_chunk[i]
                    sentence_to_keep_size = self._count_tokens(sentence_to_keep)
                    
                    if overlap_size + sentence_to_keep_size <= self.chunk_overlap:
                        overlap_sentences.insert(0, sentence_to_keep)
                        overlap_size += sentence_to_keep_size
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class SemanticSplitter(BaseTextSplitter):
    """Semantic-aware chunking that preserves meaning."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    _paragraph_pattern: Pattern[str] = re.compile(r'\n\s*\n')
    _section_pattern: Pattern[str] = re.compile(
        r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$',
        re.MULTILINE,
    )
    _list_pattern: Pattern[str] = re.compile(
        r'^\s*[-*•]\s+|^\s*\d+\.\s+',
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize SemanticSplitter.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """
        Identify semantic boundaries in text.

        Args:
            text: Input text string.

        Returns:
            Sorted list of boundary indices.
        """
        boundaries: List[int] = [0]

        for match in self._paragraph_pattern.finditer(text):
            boundaries.append(match.end())

        for match in self._section_pattern.finditer(text):
            boundaries.append(match.start())

        for match in self._list_pattern.finditer(text):
            boundaries.append(match.start())

        boundaries.append(len(text))
        return sorted(set(boundaries))


    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Split text using semantic content boundaries.

        Args:
            text: Raw input text.
            metadata: Optional metadata dict.

        Returns:
            List of semantic text chunks.
        """
        try:
            boundaries = self._find_semantic_boundaries(text)
            chunks: List[str] = []
            current_chunk = ""
            current_start = 0

            for boundary in boundaries[1:]:
                segment = text[current_start:boundary].strip()

                if len(current_chunk) + len(segment) <= self.chunk_size:
                    current_chunk += segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                        # Handle overlap
                        overlap_start = max(
                            0,
                            len(current_chunk) - self.chunk_overlap
                        )
                        current_chunk = current_chunk[overlap_start:]

                    current_chunk += segment

                current_start = boundary

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as exc:
            raise ChunkingError(f"Semantic chunking failed: {exc}") from exc


class FixedSplitter(BaseTextSplitter):
    """Fixed-size chunking with word boundary preservation."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize FixedSplitter.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap for chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Split text into fixed-size pieces preserving words.

        Args:
            text: Raw input text.
            metadata: Optional metadata.

        Returns:
            List of text chunks.
        """
        try:
            words = text.split()
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_size = 0

            for word in words:
                word_size = len(word) + 1
                if current_size + word_size > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))

                    # Overlap handling
                    overlap_words: List[str] = []
                    overlap_size = 0

                    for w in reversed(current_chunk):
                        w_size = len(w) + 1
                        if overlap_size + w_size <= self.chunk_overlap:
                            overlap_words.insert(0, w)
                            overlap_size += w_size
                        else:
                            break

                    current_chunk = overlap_words
                    current_size = overlap_size

                current_chunk.append(word)
                current_size += word_size

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        except Exception as exc:
            raise ChunkingError(f"Fixed chunking failed: {exc}") from exc


class ParagraphSplitter(BaseTextSplitter):
    """Paragraph-based chunking with size constraints."""
    __slots__: Tuple = ("chunk_size", "chunk_overlap",)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> None:
        """
        Initialize ParagraphSplitter.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Split text by paragraphs respecting size limits.

        Args:
            text: Raw input text.
            metadata: Optional metadata.

        Returns:
            List of paragraph text chunks.
        """
        try:
            paragraphs = re.split(r'\n\s*\n', text)
            chunks: List[str] = []
            current_chunk = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                        # Overlap at word-level
                        words = current_chunk.split()
                        overlap_words: List[str] = []
                        overlap_size = 0

                        for w in reversed(words):
                            w_size = len(w) + 1
                            if overlap_size + w_size <= self.chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_size += w_size
                            else:
                                break

                        current_chunk = (
                            " ".join(overlap_words) + "\n\n"
                            + paragraph
                            if overlap_words else paragraph
                        )
                    else:
                        current_chunk = paragraph

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        except Exception as exc:
            raise ChunkingError(f"Paragraph chunking failed: {exc}") from exc


# ==== ORCHESTRATION & VALIDATION MODULE ==== #
# --► STRATEGY SELECTION & STATS TRACKING

class TextChunker:
    """
    Facade for various text chunking strategies.
    
    This class orchestrates the chunking process, including text preprocessing,
    strategy selection, and post-processing validation. It is the primary
    entry point for chunking text content.
    """
    __slots__: Tuple = (
        "chunk_size", "chunk_overlap", "strategy",
        "min_chunk_size", "max_chunk_size",
        "preserve_code_blocks", "preserve_tables",
        "_splitters", "_stats", "_code_placeholder_pattern",
        "_table_placeholder_pattern"
    )

    def __init__(
        self,
        splitters: Dict[ChunkingStrategy, BaseTextSplitter],
        chunk_size: int,
        chunk_overlap: int,
        strategy: ChunkingStrategy,
        min_chunk_size: int,
        max_chunk_size: Optional[int],
        preserve_code_blocks: bool,
        preserve_tables: bool,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or chunk_size * 2
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        
        self._splitters = splitters
        self._stats: Dict[str, Any] = {
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "strategy_usage": {
                strat.value: 0 for strat in ChunkingStrategy
            },
        }
        
        self._code_placeholder_pattern = re.compile(r"(__CODE_BLOCK_\d+__)")
        self._table_placeholder_pattern = re.compile(r"(__TABLE_\d+__)")

    @classmethod
    async def create(
        cls,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 50,
        max_chunk_size: Optional[int] = None,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
    ) -> "TextChunker":
        """
        Asynchronously create a TextChunker instance.

        This factory method ensures that all underlying chunker strategies
        are initialized in a non-blocking way.
        """
        sentence_splitter = await SentenceSplitter.create(chunk_size, chunk_overlap)
        
        splitters = {
            ChunkingStrategy.SENTENCE: sentence_splitter,
            ChunkingStrategy.SEMANTIC: SemanticSplitter(chunk_size, chunk_overlap),
            ChunkingStrategy.FIXED: FixedSplitter(chunk_size, chunk_overlap),
            ChunkingStrategy.PARAGRAPH: ParagraphSplitter(chunk_size, chunk_overlap),
        }
        splitters[ChunkingStrategy.HYBRID] = splitters[strategy]

        return cls(
            splitters=splitters,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            preserve_code_blocks=preserve_code_blocks,
            preserve_tables=preserve_tables,
        )

    def _preprocess_text(
        self,
        text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess text and extract special regions (code, tables).

        Args:
            text: Raw document text.

        Returns:
            Tuple of processed text and regions metadata.

        Assumptions:
            preserve_code_blocks and preserve_tables flags are honored.
        """
        special_regions: Dict[str, Any] = {
            "code_blocks": [],
            "tables": [],
            "equations": []
        }

        processed_text = text

        if self.preserve_code_blocks:
            code_pattern = re.compile(r'```[\s\S]*?```|`[^`\n]+`',
                                      re.MULTILINE)
            for match in code_pattern.finditer(text):
                special_regions["code_blocks"].append({
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(),
                })

        if self.preserve_tables:
            table_pattern = re.compile(
                r'\|.*?\|.*?\n(?:\|[-:]+\|.*?\n)?(?:\|.*?\|.*?\n)*',
                re.MULTILINE,
            )
            for match in table_pattern.finditer(text):
                special_regions["tables"].append({
                    "start": match.start(),
                    "end": match.end(),
                    "content": match.group(),
                })

        return processed_text, special_regions

    def _select_strategy(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingStrategy:
        """
        Automatically select the best chunking strategy.

        Args:
            text: Preprocessed document text.
            metadata: Optional metadata dict.

        Returns:
            Selected ChunkingStrategy.
        """
        if self.strategy != ChunkingStrategy.HYBRID:
            return self.strategy

        has_sections = bool(
            re.search(r'\n#+\s+.*?\n|^\s*\d+\.\s+.*?$',
                      text, re.MULTILINE)
        )
        has_paragraphs = '\n\n' in text
        has_lists = bool(
            re.search(r'^\s*[-*•]\s+|^\s*\d+\.\s+',
                      text, re.MULTILINE)
        )
        avg_sentence_length = (
            len(text) /
            max(1, text.count('.') + text.count('!') + text.count('?'))
        )

        if has_sections and has_paragraphs:
            return ChunkingStrategy.SEMANTIC
        elif has_paragraphs and avg_sentence_length > 100:
            return ChunkingStrategy.PARAGRAPH
        elif avg_sentence_length < 200:
            return ChunkingStrategy.SENTENCE

        return ChunkingStrategy.FIXED

    def _validate_chunks(
        self,
        chunks: List[str],
        orig_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Validate, filter, and split oversized chunks iteratively.

        Args:
            chunks: Initial list of text chunks.
            orig_metadata: Original metadata for context.

        Returns:
            List of validated text chunks.
        """
        valid: List[str] = []
        stack = deque(chunks)

        while stack:
            chunk = stack.pop()
            text = chunk.strip()

            if len(text) < self.min_chunk_size:
                continue

            if not text or text.isspace():
                continue

            if len(text) > self.max_chunk_size:
                # Re-split oversized chunks
                sub_splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                sub_chunks = sub_splitter.split_text(text)

                if not sub_chunks or all(len(c) >= len(text) for c in sub_chunks):
                    # Fall back to hard splitting if the splitter didn't work
                    step = self.max_chunk_size - self.chunk_overlap
                    hard_chunks = [
                        text[i : i + self.max_chunk_size]
                        for i in range(0, len(text), step)
                    ]
                    sub_chunks = hard_chunks

                stack.extend(sub_chunks)
                continue

            # Check alpha ratio to filter out non-text content
            alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
            if alpha_ratio < 0.1:
                continue

            valid.append(text)

        return valid

    def split(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks using the configured strategy.

        Args:
            documents: List of Document instances to chunk.

        Returns:
            List of Document chunks.
        """
        all_documents: List[Document] = []

        for doc in documents:
            try:
                text, regions = self._preprocess_text(doc.text)

                strategy = self._select_strategy(text, doc.metadata)
                splitter = self._splitters[strategy]

                raw_chunks = splitter.split_text(text, doc.metadata)
                valid_chunks = self._validate_chunks(raw_chunks, doc.metadata)

                self._stats["strategy_usage"][strategy.value] += 1
                self._stats["total_chunks"] += len(valid_chunks)

                # Create Document objects from chunks
                for i, chunk in enumerate(valid_chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "parent_id": doc.id,
                        "chunk_index": i,
                        "chunking_strategy": strategy.value,
                        "source_length": len(doc.text),
                    })
                    
                    all_documents.append(Document(
                        text=chunk,
                        metadata=chunk_metadata
                    ))
                
                logger.debug(
                    f"Document {doc.id} produced {len(valid_chunks)} valid chunks "
                    f"via {strategy.value} strategy"
                )

            except Exception as exc:
                logger.error(f"Failed to chunk document: {exc}")
                continue

        self._stats["documents_processed"] += len(documents)

        if all_documents:
            total_size = sum(len(doc.text) for doc in all_documents)
            self._stats["avg_chunk_size"] = total_size / len(all_documents)

        logger.info(f"Processed {len(documents)} docs into "
                   f"{len(all_documents)} chunks")
        return all_documents

    @property
    def stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        return self._stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve chunking performance statistics.

        Returns:
            Dictionary containing processing stats and config.
        """
        stats_copy = self._stats.copy()
        stats_copy["config"] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.strategy.value,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }
        return stats_copy

    def reset_stats(self) -> None:
        """
        Reset performance statistics to initial state.
        """
        self._stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "strategy_usage": {
                strat.value: 0 for strat in ChunkingStrategy
            },
        }
        logger.info("Chunking statistics reset")

    def __repr__(self) -> str:
        """
        Human-readable representation of TextChunker instance.

        Returns:
            String representation including size, overlap, and strategy.
        """
        return (
            f"TextChunker(size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"strategy={self.strategy.value})"
        )