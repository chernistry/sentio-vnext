"""
Jina AI embedding provider.

This module provides an implementation of the BaseEmbedder interface for
Jina AI embeddings. It supports both synchronous and asynchronous embedding
with caching.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import requests

from src.core.embeddings.base import BaseEmbedder, EmbeddingError, _retry
from src.utils.settings import settings

logger = logging.getLogger(__name__)

# Known dimensions for Jina models
KNOWN_MODEL_DIMENSIONS = {
    "jina-embeddings-v3": 768,
    "jina-embeddings-v3": 512,
    "jina-embeddings-v3": 768,
    "jina-embeddings-v3": 512,
    "jina-embeddings-v3": 1024,
}


class JinaEmbedder(BaseEmbedder):
    """Embedding model using the Jina AI API."""

    BASE_URL: str = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        batch_size: int = 100,
        allow_empty_api_key: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Jina AI embedding client.

        Args:
            model_name: Jina embedding model to use.
            api_key: API key for Jina AI.
            max_retries: Max retry attempts.
            timeout: Request timeout in seconds.
            batch_size: Max texts per request batch.
            allow_empty_api_key: Allow missing API key (offline/dev mode).
            **kwargs: Additional arguments passed to BaseEmbedder.
        """
        self.api_key = api_key or settings.embedding_model_api_key
        if not self.api_key and not allow_empty_api_key:
            raise EmbeddingError("Jina API key not provided")

        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.allow_empty_api_key = allow_empty_api_key

        super().__init__(
            model_name=model_name,
            **kwargs,
        )

    def _get_embedding_dimension(self) -> int:
        """
        Determine embedding dimension via known map or runtime probe.
        
        Returns:
            The dimension of the embedding vectors.
            
        Raises:
            EmbeddingError: If the dimension cannot be determined.
        """
        if self.model_name in KNOWN_MODEL_DIMENSIONS:
            logger.info(
                "Using known dimension for %s: %d",
                self.model_name,
                KNOWN_MODEL_DIMENSIONS[self.model_name],
            )
            return KNOWN_MODEL_DIMENSIONS[self.model_name]

        logger.warning("Unknown dimension for model %s; probing", self.model_name)
        loop = asyncio.get_event_loop()
        test_embedding = loop.run_until_complete(self.embed_async_single("test"))
        return len(test_embedding)

    @_retry(3)
    async def _execute_async_request(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform API request with retries and error handling.
        
        Args:
            payload: Request payload to send to the API.
            
        Returns:
            List of response data items.
            
        Raises:
            EmbeddingError: If the request fails after retries.
        """
        if not self.api_key and not self.allow_empty_api_key:
            raise EmbeddingError("Jina API key not provided")

        # In test mode with allow_empty_api_key=True, return mock data
        if not self.api_key and self.allow_empty_api_key:
            # Return a mock vector for each input text
            inputs = payload.get("input", [])
            if not isinstance(inputs, list):
                inputs = [inputs]
            
            # Create mock embeddings with the correct dimension
            dimension = KNOWN_MODEL_DIMENSIONS.get(self.model_name, 1024)
            results = []
            for text in inputs:
                # Create a deterministic vector based on the hash of the text
                seed = abs(hash(text)) % 10000
                import random
                random.seed(seed)
                
                vec = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
                
                results.append({"embedding": vec})
            return results

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["data"]

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts via a single batched API call.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If the embedding fails.
        """
        if not texts:
            return []

        payload = {
            "input": texts,
            "model": self.model_name,
        }

        try:
            results = await self._execute_async_request(payload)
            return [item["embedding"] for item in results]
        except Exception as e:
            raise EmbeddingError(f"Jina embedding failed: {e}") from e

    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts asynchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If the embedding fails.
        """
        start_time = time.time()
        
        if not texts:
            return []
        
        # Check cache first
        cached_results = [self._check_cache(text) for text in texts]
        if all(cached is not None for cached in cached_results):
            self._update_stats(hit=True, duration=time.time() - start_time)
            return [c for c in cached_results if c is not None]  # type: ignore

        # Identify uncached texts
        uncached_indices = [i for i, c in enumerate(cached_results) if c is None]
        uncached_texts = [texts[i] for i in uncached_indices]

        # Split into batches to obey API limits
        batched_embeddings: List[List[float]] = []
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i : i + self.batch_size]
            batched_embeddings.extend(await self._get_embeddings(batch))

        # Merge cached + fresh embeddings
        result: List[List[float]] = []
        fresh_idx = 0
        for i, cached in enumerate(cached_results):
            if cached is not None:
                result.append(cached)
            else:
                emb = batched_embeddings[fresh_idx]
                result.append(emb)
                self._store_cache(texts[i], emb)
                fresh_idx += 1

        self._update_stats(duration=time.time() - start_time)
        return result

    async def embed_async_single(self, text: str) -> List[float]:
        """
        Get embedding for a single text asynchronously.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
            
        Raises:
            EmbeddingError: If the embedding fails.
        """
        start_time = time.time()

        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True, duration=time.time() - start_time)
            return cached

        payload = {"input": [text], "model": self.model_name}
        try:
            result = await self._execute_async_request(payload)
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            raise EmbeddingError(f"Jina embedding failed: {e}") from e

        embedding = result[0]["embedding"]
        self._store_cache(text, embedding)
        self._update_stats(duration=time.time() - start_time)
        return embedding
            
    def embed_sync(self, text: str) -> List[float]:
        """
        Get embedding for a single text synchronously.
        
        This method overrides the base implementation with a more efficient
        direct HTTP request instead of wrapping the async method.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
            
        Raises:
            EmbeddingError: If the embedding fails.
        """
        start_time = time.time()
        
        # Check cache first
        cached = self._check_cache(text)
        if cached is not None:
            self._update_stats(hit=True, duration=time.time() - start_time)
            return cached
            
        try:
            # Direct synchronous HTTP request
            payload = {
                "input": [text],
                "model": self.model_name,
            }
            
            # In test mode with allow_empty_api_key=True, return mock data
            if not self.api_key and self.allow_empty_api_key:
                # Create a deterministic vector based on hash of text
                dimension = KNOWN_MODEL_DIMENSIONS.get(self.model_name, 1024)
                seed = abs(hash(text)) % 10000
                import random
                random.seed(seed)
                embedding = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
            else:
                # Actual API call
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result["data"][0]["embedding"]
            
            # Cache the result
            self._store_cache(text, embedding)
            
            self._update_stats(duration=time.time() - start_time)
            return embedding
        except Exception as e:
            self._update_stats(error=True, duration=time.time() - start_time)
            logger.error(f"Failed to get embedding synchronously: {e}")
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    async def close(self) -> None:
        """
        Close any resources held by the embedding model.
        
        No persistent resources to close - method kept for interface symmetry.
        """
        # No-op: HTTP client is scoped per request
        pass 