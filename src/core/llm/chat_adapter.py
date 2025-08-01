"""Generic OpenAI-compatible chat adapter.

This module provides a thin wrapper around any LLM endpoint that implements
(OpenAI) `/chat/completions` semantics (OpenRouter, Groq, Together, etc.).
It supports both standard JSON responses and streaming event-source format
(`data: ...\n\n`).

Usage
-----
>>> from src.core.llm.chat_adapter import ChatAdapter
>>> adapter = ChatAdapter()
>>> resp = await adapter.chat_completion({"messages": [...], "stream": False})

The active endpoint, model and API key are taken from :pydata:`utils.settings`.
"""

import logging
from typing import Any, AsyncGenerator, Dict, Union, Optional

from src.utils.settings import settings
from src.core.llm.providers import get_provider, BaseLLMProvider

logger = logging.getLogger(__name__)

__all__ = ["ChatAdapter", "chat_completion"]


class ChatAdapter:
    """OpenAI-compatible chat client with optional streaming."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        *,
        enable_header_stealth: Optional[bool] = None,
    ) -> None:
        """Initialize the chat adapter.
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            provider: Provider name (default: from settings)
            enable_header_stealth: Whether to use stealth headers
        """
        self.provider_name = provider or settings.llm_provider
        
        # Create provider instance
        self._provider = get_provider(
            self.provider_name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_header_stealth=enable_header_stealth,
        )
        
        logger.debug("Initialized ChatAdapter with provider=%s", self.provider_name)

    async def chat_completion(
        self,
        payload: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate a chat completion.
        
        Args:
            payload: Chat completion payload
            
        Returns:
            Chat completion response or streaming generator
        """
        return await self._provider.chat_completion(payload)

    async def close(self) -> None:
        """Close the provider."""
        await self._provider.close()


# Convenience module-level facade -------------------------------------------

_adapter: Optional[ChatAdapter] = None


async def chat_completion(
    payload: Dict[str, Any],
) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
    """Convenience facade that re-uses a singleton ``ChatAdapter``."""
    global _adapter
    if _adapter is None:
        _adapter = ChatAdapter()
    return await _adapter.chat_completion(payload) 