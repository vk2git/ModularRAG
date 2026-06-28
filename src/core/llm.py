# Backwards compatibility — imports redirected to new location
# Use src.core.components.llm instead
from src.core.components.llm import LLMFactory

__all__ = ["LLMFactory"]