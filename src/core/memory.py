# Backwards compatibility — imports redirected to new location
# Use src.core.components.memory instead
from src.core.components.memory import MemoryFactory, LocalFileMessageHistory

__all__ = ["MemoryFactory", "LocalFileMessageHistory"]
