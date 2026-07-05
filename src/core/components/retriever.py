"""
Retriever factory - provides different retrieval strategies.

Strategies:
- basic: Simple top-K vector similarity search
- hybrid: Combines vector similarity + BM25 keyword search with Reciprocal Rank Fusion
- reranked: Any retriever + cross-encoder reranking pass
"""

from src.utils.config_loader import load_config
from typing import List, Optional
from langchain_core.documents import Document


class RetrieverFactory:
    """Creates retriever instances based on config."""

    def __init__(self, vector_store, embeddings=None):
        self.config = load_config()
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.retriever_config = self.config.get("retriever", {})

    def create_retriever(self, strategy: str = None, k: int = None):
        """
        Create a retriever with the given strategy.

        Args:
            strategy: 'basic', 'hybrid', or 'reranked'. Defaults to config value.
            k: Number of documents to retrieve. Defaults to config value.
        """
        strategy = strategy or self.retriever_config.get("strategy", "basic")
        k = k or self.retriever_config.get("k", 5)

        if strategy == "basic":
            return self._create_basic_retriever(k)
        elif strategy == "hybrid":
            return self._create_hybrid_retriever(k)
        elif strategy == "reranked":
            return self._create_reranked_retriever(k)
        else:
            raise ValueError(f"Unsupported retriever strategy: {strategy}")

    def _create_basic_retriever(self, k: int):
        """Simple vector similarity retriever."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def _create_hybrid_retriever(self, k: int):
        """
        Hybrid retriever combining vector similarity + BM25 keyword search.
        Uses Reciprocal Rank Fusion to merge results.
        """
        from langchain.retrievers import EnsembleRetriever

        # Vector retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # BM25 keyword retriever
        try:
            from langchain_community.retrievers import BM25Retriever

            # Get all documents from the vector store for BM25
            # This is a one-time operation; for large collections consider caching
            all_docs = self.vector_store.similarity_search("", k=100)
            if all_docs:
                bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
            else:
                print("⚠️  No documents found for BM25 retriever, falling back to vector-only")
                return vector_retriever

        except ImportError:
            print("⚠️  BM25 not available (install rank-bm25), falling back to vector-only")
            return vector_retriever

        weights = self.retriever_config.get("hybrid_weights", [0.5, 0.5])

        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights,
        )

    def _create_reranked_retriever(self, k: int):
        """
        Retriever with cross-encoder reranking.
        First retrieves more docs, then reranks to top-K.
        """
        from src.core.components.reranker import RerankerFactory

        # Retrieve more candidates than needed, then rerank
        candidate_k = k * 3
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": candidate_k})

        reranker = RerankerFactory()
        compressor = reranker.create_reranker()

        from langchain.retrievers import ContextualCompressionRetriever

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
