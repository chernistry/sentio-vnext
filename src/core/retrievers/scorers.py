from __future__ import annotations

"""Scoring plugins for document retrieval.

This module provides various scorer plugins that can be used with retrievers
to enhance document ranking based on different signals.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Pattern

from src.core.models.document import Document
from src.core.retrievers.base import ScorerPlugin

logger = logging.getLogger(__name__)


class KeywordMatchScorer(ScorerPlugin):
    """Score documents based on keyword match counts.
    
    This scorer counts exact matches of keywords in the document text
    and produces a normalized score based on match density.
    
    Args:
        weight: Weight of the keyword score in the overall ranking
        case_sensitive: Whether to perform case-sensitive matching
    """
    
    def __init__(self, weight: float = 0.5, case_sensitive: bool = False):
        self.weight = weight
        self.case_sensitive = case_sensitive
        
    def score(self, query: str, docs: List[Document]) -> List[float]:
        """Score documents by counting keyword matches.
        
        Args:
            query: The query string to extract keywords from
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        # Extract keywords (simple approach - split on whitespace and punctuation)
        keywords = set(re.findall(r'\w+', query.lower()))
        if not keywords:
            return [0.0] * len(docs)
        
        # Prepare document texts based on case sensitivity
        doc_texts = []
        for doc in docs:
            if self.case_sensitive:
                doc_texts.append(doc.text)
            else:
                doc_texts.append(doc.text.lower())
                
        # Count keyword matches in each document
        scores = []
        for text in doc_texts:
            text_words = set(re.findall(r'\w+', text))
            matches = len(keywords.intersection(text_words))
            # Normalize by keyword count and apply weight
            score = (matches / len(keywords)) * self.weight
            scores.append(score)
            
        return scores


class RecencyScorer(ScorerPlugin):
    """Score documents based on recency information in metadata.
    
    This scorer assumes documents have timestamp information in metadata
    and assigns higher scores to more recent documents.
    
    Args:
        timestamp_field: Name of the metadata field containing timestamp
        weight: Weight of recency score in the overall ranking
        max_age_seconds: Maximum document age in seconds for normalization
    """
    
    def __init__(
        self, 
        timestamp_field: str = "timestamp", 
        weight: float = 0.3, 
        max_age_seconds: int = 86400 * 30  # 30 days
    ):
        self.timestamp_field = timestamp_field
        self.weight = weight
        self.max_age_seconds = max_age_seconds
        
    def score(self, query: str, docs: List[Document]) -> List[float]:
        """Score documents based on recency.
        
        Args:
            query: The query string (not used in this scorer)
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        import time
        
        current_time = time.time()
        scores = []
        
        for doc in docs:
            # Default score is 0
            score = 0.0
            
            # Extract timestamp from metadata if available
            timestamp = doc.metadata.get(self.timestamp_field)
            if timestamp and isinstance(timestamp, (int, float)):
                age_seconds = current_time - timestamp
                if age_seconds >= 0:
                    # Newer documents get higher scores
                    # Normalize to 0-1 range based on max_age
                    normalized_age = min(age_seconds, self.max_age_seconds) / self.max_age_seconds
                    score = (1 - normalized_age) * self.weight
                    
            scores.append(score)
            
        return scores


class SemanticSimilarityScorer(ScorerPlugin):
    """Score documents based on semantic similarity to query.
    
    This scorer uses a text embedding model to compute semantic similarity
    between the query and documents.
    
    Args:
        embedder: The embedding model to use
        weight: Weight of semantic score in the overall ranking
    """
    
    def __init__(self, embedder, weight: float = 0.7):
        """Initialize the semantic scorer.
        
        Args:
            embedder: Any embedding model with an embed_sync method
            weight: Weight of semantic score in the overall ranking
        """
        self.embedder = embedder
        self.weight = weight
        
    def score(self, query: str, docs: List[Document]) -> List[float]:
        """Score documents based on semantic similarity.
        
        Args:
            query: The query to compare documents against
            docs: List of documents to score
            
        Returns:
            List of scores corresponding to each document
        """
        try:
            import numpy as np
            
            # Embed query
            query_embedding = self.embedder.embed_sync(query)
            
            # Embed document texts
            doc_texts = [doc.text for doc in docs]
            if not doc_texts:
                return []
                
            doc_embeddings = self.embedder.embed_batch_sync(doc_texts)
            
            # Calculate cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            scores = []
            
            for doc_emb in doc_embeddings:
                doc_norm = np.linalg.norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                    scores.append(float(similarity * self.weight))
                else:
                    scores.append(0.0)
                    
            return scores
            
        except Exception as e:
            logger.warning("Error in semantic scoring: %s", e)
            return [0.0] * len(docs) 