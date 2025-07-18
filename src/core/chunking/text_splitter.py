"""
Advanced text chunking with multiple strategies and optimization.

This module provides tools for splitting documents into smaller chunks
with configurable chunk size and overlap.
"""

import logging
import re
import asyncio

# NEW: Optional LangChain & LlamaIndex imports
try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
    _LC_SPLITTERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _LC_SPLITTERS_AVAILABLE = False
    CharacterTextSplitter = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    SemanticChunker = None  # type: ignore

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
    # Recursive structure-aware splitting similar to LangChain's
    RECURSIVE = "recursive"


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

# NEW: Adapter to wrap LangChain splitters into our BaseTextSplitter API
class _LCTextSplitterAdapter(BaseTextSplitter):
    """Wrap external LangChain/LlamaIndex splitter to provide split_text("""
    __slots__: Tuple = ("_inner",)

    def __init__(self, inner_splitter: Any) -> None:
        self._inner = inner_splitter

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not text or not text.strip():
            return []
        # Delegate to external splitter
        if hasattr(self._inner, "split_text"):
            chunks = self._inner.split_text(text)  # type: ignore
        elif hasattr(self._inner, "create_documents"):
            docs = self._inner.create_documents([text])  # type: ignore
            chunks = [d.page_content for d in docs]
        else:
            raise ChunkingError("Unsupported splitter interface")
        return [c.strip() for c in chunks if c and c.strip()]


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
        Count the approximate number of *characters* in text.

        Note: Although the method name says "tokens" for backward compatibility,
        our most common use-cases (and test-suite) expect *character* budgeting.
        """
        return len(text)

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

        chunks: List[str] = []
        start_idx = 0
        text_len = len(text)
        
        while start_idx < text_len:
            # Determine the end of the next chunk
            end_idx = min(start_idx + self.chunk_size, text_len)
            
            # Find a natural break point (e.g., end of a sentence or word)
            # to avoid splitting mid-word.
            if end_idx < text_len:
                # Look for sentence boundaries first
                sentence_break = text.rfind('.', start_idx, end_idx)
                if sentence_break != -1:
                    end_idx = sentence_break + 1
                else:
                    # If no sentence break, look for a space
                    space_break = text.rfind(' ', start_idx, end_idx)
                    if space_break != -1:
                        end_idx = space_break
            
            chunk = text[start_idx:end_idx].strip()
            
            if chunk:
                chunks.append(chunk)

            # Move to the start of the next chunk, considering overlap
            start_idx = end_idx - self.chunk_overlap
            if start_idx < end_idx - self.chunk_overlap + self.chunk_overlap and len(chunks) > 1:
                 start_idx = end_idx

            if end_idx >= text_len:
                break
        
        # In case of a single long string with no natural breaks
        if not chunks and text.strip():
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size
            chunks = [text[i: i + self.chunk_size] for i in range(0, len(text), step)]

        return [c for c in chunks if c]


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


# ==== RECURSIVE STRUCTURE-AWARE SPLITTER ====


class RecursiveSplitter(BaseTextSplitter):
    """Recursive, structure-aware splitter mirrored on LangChain's RecursiveCharacterTextSplitter.

    The splitter attempts to keep larger semantic units (paragraphs, sentences, words)
    intact by recursively descending through a list of separators until chunks fit
    the desired size. This generally yields higher-quality chunks than naïve
    character splitting while remaining fast and memory-efficient.
    """

    __slots__: Tuple = (
        "chunk_size",
        "chunk_overlap",
        "separators",
    )

    _DEFAULT_SEPARATORS: List[str] = [
        "\n\n",  # Paragraph break
        "\n",    # Line break / sentence-ish
        " ",     # Word boundary
        "",       # Fallback to character level
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self._DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, level: int = 0) -> List[str]:
        """Recursively split *text* using the *level*-th separator."""
        if len(text) <= self.chunk_size or level >= len(self.separators):
            # Base case 1 – text is already small enough
            # Base case 2 – no more separators: hard split
            step = self.chunk_size - self.chunk_overlap if self.chunk_overlap else self.chunk_size
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        sep = self.separators[level]
        if sep:
            parts = text.split(sep)
            # Re-append separator to preserve context except for last part
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
        else:
            # Empty separator → character-level splitting
            parts = list(text)

        chunks: List[str] = []
        for part in parts:
            # Recurse if piece still too large
            if len(part) > self.chunk_size:
                chunks.extend(self._recursive_split(part, level + 1))
            else:
                chunks.append(part)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if not self.chunk_overlap or not chunks:
            return chunks

        with_overlap: List[str] = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev_tail = chunks[idx - 1][-self.chunk_overlap :]
            with_overlap.append(prev_tail + chunks[idx])
        return with_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Split *text* into size-bounded chunks using recursive rules."""
        if not text or not text.strip():
            return []

        initial_chunks = self._recursive_split(text)
        return self._apply_overlap(initial_chunks)


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
        # Base internal splitter
        sentence_splitter_internal = await SentenceSplitter.create(
            chunk_size, chunk_overlap
        )
        # Choose external vs internal implementations
        if _LC_SPLITTERS_AVAILABLE:
            # Prepare LangChain splitters
            _lc_recursive = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            _lc_sentence = _lc_recursive
            _lc_fixed = CharacterTextSplitter(
                separator=" ",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            _lc_paragraph = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # Semantic uses recursive splitter by default
            _lc_semantic = _lc_recursive
            splitters: Dict[ChunkingStrategy, BaseTextSplitter] = {
                ChunkingStrategy.SENTENCE: _LCTextSplitterAdapter(_lc_sentence),
                ChunkingStrategy.SEMANTIC: _LCTextSplitterAdapter(_lc_semantic),
                ChunkingStrategy.FIXED: _LCTextSplitterAdapter(_lc_fixed),
                ChunkingStrategy.PARAGRAPH: _LCTextSplitterAdapter(_lc_paragraph),
                ChunkingStrategy.RECURSIVE: _LCTextSplitterAdapter(_lc_recursive),
                ChunkingStrategy.HYBRID: _LCTextSplitterAdapter(_lc_sentence),
            }
        else:
            # Fallback to internal implementations
            splitters = {
                ChunkingStrategy.SENTENCE: sentence_splitter_internal,
                ChunkingStrategy.SEMANTIC: SemanticSplitter(chunk_size, chunk_overlap),
                ChunkingStrategy.FIXED: FixedSplitter(chunk_size, chunk_overlap),
                ChunkingStrategy.PARAGRAPH: ParagraphSplitter(chunk_size, chunk_overlap),
                ChunkingStrategy.RECURSIVE: RecursiveSplitter(chunk_size, chunk_overlap),
                ChunkingStrategy.HYBRID: sentence_splitter_internal,
            }

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