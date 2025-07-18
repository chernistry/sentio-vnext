"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, AsyncGenerator, Optional


class BaseLLMProvider(ABC):
    """Base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the provider."""
        pass 