# Backwards compatibility — imports redirected to new location
# Use src.core.components.vector_store instead
from src.core.components.vector_store import VectorStoreFactory

__all__ = ["VectorStoreFactory"]