from src.utils.config_loader import load_config
from src.utils.class_loader import instantiate_class
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
            if not api_key:
                raise ValueError("PINECONE_API_KEY is not found in environment variables")
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=api_key
            )
        elif provider == "custom":
            custom_conf = self.db_config.get("custom", {})
            module_path = custom_conf.get("module_path")
            class_name = custom_conf.get("class_name")
            kwargs = custom_conf.get("kwargs", {})
            
            if not module_path or not class_name:
                raise ValueError("Custom Vector Store config requires 'module_path' and 'class_name'")
                
            print(f"---- Initializing Custom Vector Store: {class_name} from {module_path}")
            return instantiate_class(module_path, class_name, embedding=embeddings, **kwargs)

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