from src.utils.config_loader import load_config
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import json
from typing import List
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

class LocalFileMessageHistory(BaseChatMessageHistory):
    """
    A simple file-based chat history that persists to a JSON file.
    """
    def __init__(self, session_id: str, file_path: str, k: int = 10):
        self.session_id = session_id
        self.file_path = file_path
        self.k = k
        self.messages = []
        self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            return
        
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                session_data = data.get(self.session_id, [])
                self.messages = messages_from_dict(session_data)
                
                if len(self.messages) > self.k:
                     self.messages = self.messages[-self.k:]
        except (json.JSONDecodeError, FileNotFoundError):
            self.messages = []

    def _save(self):
        all_data = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        all_data[self.session_id] = messages_to_dict(self.messages)
        
        with open(self.file_path, "w") as f:
            json.dump(all_data, f, indent=4)

    def add_message(self, message: BaseMessage):
        self.messages.append(message)
        
        # Apply sliding window
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k:]
            
        self._save()

    def clear(self):
        self.messages = []
        self._save()

class MemoryFactory:
    def __init__(self):
        self.config = load_config()
        self.memory_config = self.config.get("memory", {})

    def get_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Returns a chat history object for the given session_id.
        """
        provider = self.memory_config.get("provider", "local").lower()
        
        if provider == "local":
            return self._create_local_history(session_id)
        elif provider == "custom":
            return self._create_custom_history(session_id)
        else:
             raise ValueError(f"Unsupported memory provider: {provider}")

    def _create_local_history(self, session_id: str) -> LocalFileMessageHistory:
        file_path = self.memory_config.get("local", {}).get("file_path", "chat_history.json")
        k = self.memory_config.get("window_size", 10)
        return LocalFileMessageHistory(session_id, file_path, k=k)

    def _create_redis_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            # The import is already at the top, but keeping this structure for robustness
            from langchain_community.chat_message_histories import RedisChatMessageHistory
        except ImportError:
            raise ImportError("Redis support requires 'redis' package. Please install it.")
            
        redis_url = self.memory_config.get("redis", {}).get("url", "redis://localhost:6379")
        ttl = self.memory_config.get("redis", {}).get("ttl", 86400) # Default 24h TTL
        
        print(f"Connecting to Redis Memory at {redis_url} (Session: {session_id})")
        return RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            ttl=ttl
        )

    def _create_custom_history(self, session_id: str) -> BaseChatMessageHistory:
        from src.utils.class_loader import instantiate_class
        
        custom_conf = self.memory_config.get("custom", {})
        module_path = custom_conf.get("module_path")
        class_name = custom_conf.get("class_name")
        kwargs = custom_conf.get("kwargs", {})
        
        if not module_path or not class_name:
            raise ValueError("Custom Memory config requires 'module_path' and 'class_name'")
            
        print(f"Initializing Custom Memory: {class_name} from {module_path} (Session: {session_id})")
        # Most memory classes take session_id as first arg or kwargs
        # We pass it as kwarg 'session_id' and also positional if needed, but kwargs is safer
        # LangChain standard is often session_id in init
        return instantiate_class(module_path, class_name, session_id=session_id, **kwargs)

    def create_vector_memory(self, vector_store, k: int = 5):
        """
        Creates a VectorStoreRetrieverMemory backed by the given vector_store.
        """
        try:
            from langchain.memory import VectorStoreRetrieverMemory
        except ImportError:
            try:
                from langchain_classic.memory import VectorStoreRetrieverMemory
            except ImportError:
                 raise ImportError("Could not import VectorStoreRetrieverMemory. Please check langchain installation.")

        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return VectorStoreRetrieverMemory(retriever=retriever)
