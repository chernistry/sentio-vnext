from __future__ import annotations

"""Centralised runtime configuration for Sentio.

This thin wrapper around ``pydantic.BaseSettings`` exposes environment
variables as a typed, cached singleton.  Other modules should import
``settings`` and use attribute access instead of reaching for ``os.environ``
directly.  This improves testability and documents expected configuration
keys in one place.

Example
-------
>>> from root.src.utils.settings import settings
>>> print(settings.embedding_provider)
'jina'

The class deliberately **does not** hard-fail on unknown variables — that keeps
backward-compat and allows plugins to read custom env vars on their own.
"""

import functools
from typing import Optional, ClassVar
import os

# Detect major Pydantic version at import-time to avoid incompatible config declarations.
try:
    import pydantic  # type: ignore

    _IS_PYD_V2 = pydantic.version.VERSION.startswith("2")
except Exception:  # pragma: no cover – very defensive
    _IS_PYD_V2 = False  # Assume v1 if detection fails

# ---------------------------------------------------------------------------
# Compatibility shim for *pydantic* v1 → v2 migration.
# ---------------------------------------------------------------------------
try:
    # Pydantic < 2.0
    from pydantic import BaseSettings, Field  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback for v2+
    from pydantic_settings import BaseSettings  # type: ignore
    from pydantic import Field  # Field still exists in v2

# Import SettingsConfigDict at module level (optional dependency, only used when
# running on pydantic>=2). Importing here avoids leaking the symbol into the
# class namespace which would otherwise be interpreted as a model field.
try:
    from pydantic_settings import SettingsConfigDict  # type: ignore
except ImportError:  # pragma: no cover – optional for pydantic v1
    SettingsConfigDict = None  # type: ignore


# ---------------------------------------------------------------------------
# Model definition – conditional *Config* vs *model_config* for Pydantic v1/v2
# ---------------------------------------------------------------------------


class SentioSettings(BaseSettings):
    """Environment-driven settings for the whole application."""

    log_level: str = Field("info", env="LOG_LEVEL")

    # Embeddings
    embedding_provider: str = Field("jina", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("jina-embeddings-v3", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(16, env="EMBEDDING_BATCH_SIZE")

    # Embedding provider API key (e.g., Jina AI)
    embedding_model_api_key: Optional[str] = Field(None, env="EMBEDDING_MODEL_API_KEY")

    # Generic OpenAI-compatible chat LLM configuration
    chat_provider: str = Field("openai", env="CHAT_PROVIDER")
    chat_llm_base_url: str = Field("https://api.openai.com/v1", env="CHAT_LLM_BASE_URL")
    chat_llm_model: str = Field("gpt-4o", env="CHAT_LLM_MODEL")
    chat_llm_api_key: str | None = Field(None, env="CHAT_LLM_API_KEY")

    # Chunking --------------------------------------------------------------
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(50, env="MIN_CHUNK_SIZE")
    max_chunk_size: int = Field(1024, env="MAX_CHUNK_SIZE")
    chunking_strategy: str = Field("sentence", env="CHUNKING_STRATEGY")

    # Vector store ---------------------------------------------------------
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")

    # --------------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------------
    def chunking_kwargs(self) -> dict[str, int | str]:
        """Return dict with chunking-related configuration suitable for TextChunker.create()."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.chunking_strategy.lower(),
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    # Dynamic fallback -----------------------------------------------------
    def __getattr__(self, item: str):  # noqa: D401 – simple
        """Fallback to environment variables for unknown attributes (case-insensitive)."""
        env_val = os.getenv(item.upper())
        if env_val is not None:
            return env_val
        raise AttributeError(item)

    if not _IS_PYD_V2:
        # Pydantic < 2.0 – classic *Config* subclass.
        class Config:  # noqa: WPS431 (inner class by design)
            case_sensitive = False
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    else:
        # Pydantic ≥ 2.0 – new ``SettingsConfigDict`` mechanism.
        if SettingsConfigDict is None:
            raise RuntimeError(
                "pydantic>=2 detected but the optional 'pydantic-settings' package "
                "is missing. Install it or pin pydantic<2."
            )

        # Mark the config object as a :class:`~typing.ClassVar` so Pydantic
        # does not treat it as a regular model field.
        model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
            extra="ignore",
            case_sensitive=False,
            env_file=".env",
            env_file_encoding="utf-8",
        )

    # Convenience helpers -------------------------------------------------

    def to_common_kwargs(self) -> dict[str, str | int | bool | None]:
        """Return dict with generic embedding-constructor kwargs."""
        return {
            "model_name": self.embedding_model,
            "batch_size": self.embedding_batch_size,
            # Additional fields can be mapped as needed.
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> SentioSettings:  # pragma: no cover – deterministic
    """Return cached settings instance (singleton)."""
    return SentioSettings()  # type: ignore[return-value]


# Eagerly create default instance for import-style usage
settings: SentioSettings = get_settings()
