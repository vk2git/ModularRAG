"""
Shared modular components for all RAG architectures.

Components are the building blocks that every architecture uses:
- LLM (language model)
- Embeddings (text → vector)
- Vector Store (vector database)
- Memory (conversation history)
- Retriever (document retrieval strategies)
- Reranker (result reranking)
"""

from src.core.components.llm import LLMFactory
from src.core.components.embedding import EmbeddingFactory
from src.core.components.vector_store import VectorStoreFactory
from src.core.components.memory import MemoryFactory

__all__ = [
    "LLMFactory",
    "EmbeddingFactory",
    "VectorStoreFactory",
    "MemoryFactory",
]
