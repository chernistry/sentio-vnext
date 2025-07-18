"""LLM provider adapters.

This package contains adapters for different LLM providers.
"""

from typing import Dict, Type, Optional, Any

from .base import BaseLLMProvider

__all__ = ["get_provider", "BaseLLMProvider"]

_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {}


def register_provider(name: str):
    """Register a provider class."""
    def decorator(cls: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        _PROVIDERS[name] = cls
        return cls
    return decorator


def get_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Get a provider instance by name.
    
    Args:
        name: Provider name
        **kwargs: Additional provider-specific arguments
        
    Returns:
        A provider instance
    """
    if name not in _PROVIDERS:
        # Import built-in providers
        from . import openai  # noqa: F401
        
        # Check again after imports
        if name not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {name}")
    
    return _PROVIDERS[name](**kwargs) 