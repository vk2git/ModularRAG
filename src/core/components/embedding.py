import os
from src.utils.config_loader import load_config
class EmbeddingFactory:
    def __init__(self):
        self.config = load_config()
        self.embed_config = self.config.get("embedding", {})

    def create_embeddings_model(self):
        """
        Creates the embedding model based on config
        """
        provider = self.embed_config.get("provider", "huggingface").lower()

        if provider == "huggingface":
            model_name = self.embed_config.get("huggingface", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            print(f"---- Loading Local Embeddings: {model_name}")
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        elif provider == "openai":
            model_name = self.embed_config.get("openai", {}).get("model_name", "text-embedding-3-small")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not found in environment variables")
            print(f"---- Loading Cloud Embeddings from OpenAI: {model_name}")
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name, api_key=api_key)
        elif provider == "google":
            model_name = self.embed_config.get("google", {}).get("model_name", "models/text-embedding-004")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not found in environment variables")
            print(f"---- Loading Cloud embedding from Google: {model_name}")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        else:
            raise ValueError(f"Unsupported Embedding provider: {provider}")
