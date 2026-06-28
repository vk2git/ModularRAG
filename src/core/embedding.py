# Backwards compatibility — imports redirected to new location
# Use src.core.components.embedding instead
from src.core.components.embedding import EmbeddingFactory

__all__ = ["EmbeddingFactory"]