"""
Base class for all RAG architecture implementations.

Every architecture must inherit from BaseArchitecture and implement:
- run(query, session_id) -> str
- get_info() -> dict (static metadata)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseArchitecture(ABC):
    """
    Abstract base class for all RAG architectures.

    Each architecture represents a distinct RAG pattern (e.g., Naive, Advanced,
    Corrective, Self-RAG, Agentic, Adaptive, Graph) and implements its own
    retrieval and generation strategy using shared modular components.
    """

    # Subclasses MUST set these class attributes
    name: str = ""                  # e.g., "naive"
    display_name: str = ""          # e.g., "Naive RAG"
    description: str = ""           # One-line description
    requires: list = []             # Hard dependencies that block usage, e.g., ["langgraph"]
    optional_deps: list = []        # Soft dependencies that enhance but aren't required, e.g., ["neo4j"]

    def __init__(self, components: Dict[str, Any], config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the architecture with shared components.

        Args:
            components: Dict containing shared instances:
                - llm: Language model
                - embeddings: Embedding model
                - vector_store: Vector database
                - memory_factory: MemoryFactory instance
                - guardrails: GuardrailsManager instance
                - retriever_factory: RetrieverFactory instance (optional)
            config: Architecture-specific configuration dict
            verbose: Enable verbose logging
        """
        self.components = components
        self.config = config
        self.verbose = verbose

        # Convenience shortcuts to shared components
        self.llm = components["llm"]
        self.embeddings = components["embeddings"]
        self.vector_store = components["vector_store"]
        self.memory_factory = components["memory_factory"]
        self.guardrails = components["guardrails"]

    def _log(self, msg: str):
        """Log message only if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    @abstractmethod
    def run(self, query: str, session_id: str = "default") -> str:
        """
        Execute the RAG pipeline for this architecture.

        Args:
            query: User's question
            session_id: Session identifier for conversation history

        Returns:
            The generated response string
        """
        pass

    def get_info(self) -> Dict[str, str]:
        """Return metadata about this architecture."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "requires": self.requires,
        }

    def check_requirements(self) -> Dict[str, Any]:
        """
        Check if optional dependencies for this architecture are installed.

        Returns:
            Dict with 'available' (bool) and 'missing' (list of missing packages)
        """
        missing = []
        for req in self.requires:
            try:
                __import__(req.replace("-", "_"))
            except ImportError:
                missing.append(req)

        return {
            "available": len(missing) == 0,
            "missing": missing,
        }

    def _validate_input(self, query: str) -> Dict[str, Any]:
        """Run input through guardrails. Returns validation result dict."""
        return self.guardrails.validate_input(query)

    def _validate_output(self, response: str) -> str:
        """Run output through guardrails. Returns sanitized response."""
        output_validation = self.guardrails.validate_output(response)
        if not output_validation["valid"]:
            print(f"⚠️  Output validation warning: {output_validation['reason']}")
            return output_validation["sanitized_output"]
        return response

    def _get_chat_history(self, session_id: str):
        """Get chat history messages for the session."""
        history = self.memory_factory.get_chat_history(session_id)
        memory_type = self.config.get("memory", {}).get("type", "window")

        if memory_type == "window":
            return history, history.messages
        elif memory_type == "summary":
            try:
                from langchain.memory import ConversationSummaryBufferMemory
            except ImportError:
                from langchain_classic.memory import ConversationSummaryBufferMemory

            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=history,
                max_token_limit=1000,
                return_messages=True
            )
            return history, memory.load_memory_variables({})["history"]
        elif memory_type == "vector":
            from src.core.components.vector_store import VectorStoreFactory
            from langchain_core.messages import SystemMessage

            mem_config = self.config.get("memory", {}).get("vector_memory", {})
            collection_name = mem_config.get("collection_name", "chat_memory")
            k = mem_config.get("k", 5)

            vf = VectorStoreFactory()
            memory_store = vf.create_vector_store(self.embeddings, collection_name=collection_name)
            vector_memory = self.memory_factory.create_vector_memory(memory_store, k=k)

            memory_data = vector_memory.load_memory_variables({"prompt": ""})
            retrieved_history = memory_data.get("history", "")

            chat_history = [SystemMessage(content=f"Relevant Past Conversations:\n{retrieved_history}")]
            return history, chat_history
        else:
            return history, history.messages

    def _save_to_history(self, history, query: str, response: str):
        """Save the interaction to chat history."""
        from langchain_core.messages import HumanMessage, AIMessage
        history.add_message(HumanMessage(content=query))
        history.add_message(AIMessage(content=response))
