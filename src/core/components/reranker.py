"""
Reranker factory — provides cross-encoder reranking for improved retrieval quality.

Available rerankers:
- cross_encoder (free, local): Uses sentence-transformers cross-encoder models
- cohere (cloud, API key): Uses Cohere Rerank API

Default: cross_encoder (free, no API key needed)
To use Cohere: set reranker.provider to 'cohere' and COHERE_API_KEY env var
"""

from src.utils.config_loader import load_config
import os


class RerankerFactory:
    """Creates reranker instances based on config."""

    def __init__(self):
        self.config = load_config()
        self.reranker_config = self.config.get("reranker", {})

    def create_reranker(self):
        """
        Create a reranker/compressor based on config.

        Returns a LangChain BaseDocumentCompressor.
        """
        provider = self.reranker_config.get("provider", "cross_encoder").lower()

        if provider == "cross_encoder":
            return self._create_cross_encoder_reranker()
        elif provider == "cohere":
            return self._create_cohere_reranker()
        else:
            raise ValueError(f"Unsupported reranker provider: {provider}")

    def _create_cross_encoder_reranker(self):
        """
        Free, local cross-encoder reranker using sentence-transformers.
        Requires: pip install sentence-transformers
        """
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            from langchain.retrievers.document_compressors import CrossEncoderReranker
        except ImportError:
            raise ImportError(
                "Cross-encoder reranking requires 'sentence-transformers'. "
                "Install with: uv pip install 'modular-rag[rerank]'"
            )

        model_name = self.reranker_config.get(
            "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        top_n = self.reranker_config.get("top_n", 5)

        print(f"---- Loading Cross-Encoder Reranker: {model_name}")
        model = HuggingFaceCrossEncoder(model_name=model_name)
        return CrossEncoderReranker(model=model, top_n=top_n)

    def _create_cohere_reranker(self):
        """
        Cloud-based Cohere Rerank API.
        Requires: pip install langchain-cohere and COHERE_API_KEY env var
        """
        try:
            from langchain_cohere import CohereRerank
        except ImportError:
            raise ImportError(
                "Cohere reranking requires 'langchain-cohere'. "
                "Install with: uv pip install 'modular-rag[rerank]'"
            )

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is not found in environment variables")

        model = self.reranker_config.get("cohere_model", "rerank-english-v3.0")
        top_n = self.reranker_config.get("top_n", 5)

        print(f"---- Loading Cohere Reranker: {model}")
        return CohereRerank(
            cohere_api_key=api_key,
            model=model,
            top_n=top_n,
        )
