from src.utils.config_loader import load_config
import os

class VectorStoreFactory:
    def __init__(self):
        self.config = load_config()
        self.vector_db_config = self.config.get("vector_db", {})

    def create_vector_store(self, embeddings, collection_name: str = None):
        """
        Creates and returns a vector store instance.
        """
        provider = self.vector_db_config.get("provider", "chroma").lower()
        
        if provider == "chroma":
            return self._create_chroma_db(embeddings, collection_name)
        elif provider == "pinecone":
            return self._create_pinecone(embeddings)
        else:
            raise ValueError(f"Unsupported vector DB provider: {provider}")

    def _create_chroma_db(self, embeddings, collection_name: str = None):
        from langchain_chroma import Chroma
        
        persist_dir = self.vector_db_config.get("persist_directory", "./chroma_db")
        if not collection_name:
            collection_name = self.vector_db_config.get("collection_name", "rag_collection")
            
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )

    def _create_pinecone(self, embeddings):
        """
        Creates a Pinecone vector store instance.
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not found in environment variables")
        
        pinecone_conf = self.vector_db_config.get("pinecone", {})
        index_name = pinecone_conf.get("index_name", "rag-index")
        
        from langchain_pinecone import PineconeVectorStore
        print(f"---- Initializing Pinecone Vector Store: {index_name}")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=api_key
        )
