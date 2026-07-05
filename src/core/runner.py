"""
Pipeline Runner - unified interface to run any RAG architecture.

Handles:
1. Loading global config + architecture-specific config
2. Initializing shared components (LLM, embeddings, vector store, memory, guardrails)
3. Instantiating the selected architecture
4. Providing a unified run() interface
"""

from typing import Optional, Dict, Any
from src.core.registry import ArchitectureRegistry
from src.core.components.llm import LLMFactory
from src.core.components.embedding import EmbeddingFactory
from src.core.components.vector_store import VectorStoreFactory
from src.core.components.memory import MemoryFactory
from src.core.guardrails import GuardrailsManager
from src.utils.config_loader import load_config, load_architecture_config


class PipelineRunner:
    """
    Unified runner for any RAG architecture.

    Initializes shared components once and delegates to the selected
    architecture for query processing.

    Usage:
        runner = PipelineRunner()              # Uses config default
        runner = PipelineRunner("advanced")    # Override architecture
        response = runner.run("What is X?")
        runner.switch_architecture("corrective")  # Hot-switch
    """

    def __init__(self, architecture_name: Optional[str] = None, verbose: bool = False):
        """
        Initialize the pipeline runner.

        Args:
            architecture_name: Override the architecture from config. None = use config default.
            verbose: Enable verbose logging for all components.
        """
        self.verbose = verbose
        self.global_config = load_config()

        # Determine which architecture to use
        self.registry = ArchitectureRegistry()

        if architecture_name:
            self.architecture_name = architecture_name
        else:
            self.architecture_name = (
                self.global_config.get("architecture", {}).get("active", "naive")
            )

        self._log(f"---- Initializing ModularRAG with architecture: {self.architecture_name}")

        # Initialize shared components (done once, shared across architectures)
        self._init_components()

        # Initialize the selected architecture
        self._init_architecture()

    def _log(self, msg: str):
        """Log message only if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _init_components(self):
        """Initialize all shared components."""
        self._log("---- Loading shared components...")

        # LLM
        llm_factory = LLMFactory()
        self.llm = llm_factory.create_llm()

        # Embeddings
        embed_factory = EmbeddingFactory()
        self.embeddings = embed_factory.create_embeddings_model()

        # Vector Store
        vector_factory = VectorStoreFactory()
        self.vector_store = vector_factory.create_vector_store(self.embeddings)

        # Memory
        self.memory_factory = MemoryFactory()

        # Guardrails
        guardrails_config = self.global_config.get("guardrails", {})
        self.guardrails = GuardrailsManager(guardrails_config, llm=self.llm)

        # Bundle components for architecture use
        self.components = {
            "llm": self.llm,
            "embeddings": self.embeddings,
            "vector_store": self.vector_store,
            "memory_factory": self.memory_factory,
            "guardrails": self.guardrails,
        }

        self._log("---- All components loaded ✓")

    def _init_architecture(self):
        """Initialize the selected architecture."""
        arch_cls = self.registry.get_architecture_class(self.architecture_name)

        # Check requirements
        req_check = arch_cls(
            components=self.components,
            config={},
            verbose=self.verbose
        ).check_requirements()

        if not req_check["available"]:
            missing = ", ".join(req_check["missing"])
            print(f"⚠️  Architecture '{self.architecture_name}' has missing dependencies: {missing}")
            print(f"   Install with: uv pip install {' '.join(req_check['missing'])}")
            print(f"   Falling back to 'naive' architecture.")
            self.architecture_name = "naive"
            arch_cls = self.registry.get_architecture_class("naive")

        # Load architecture-specific config
        arch_config = load_architecture_config(self.architecture_name)

        # Merge: global config values are defaults, architecture config overrides
        merged_config = {**self.global_config, **arch_config}

        self.architecture = arch_cls(
            components=self.components,
            config=merged_config,
            verbose=self.verbose,
        )

        self._log(f"---- Architecture '{self.architecture.display_name}' initialized ✓")

    def run(self, query: str, session_id: str = "default") -> str:
        """
        Run a query through the active architecture.

        Args:
            query: User's question
            session_id: Session ID for conversation history

        Returns:
            Generated response string
        """
        return self.architecture.run(query, session_id)

    def switch_architecture(self, name: str):
        """
        Switch to a different architecture without reinitializing components.

        Args:
            name: Architecture name to switch to
        """
        if name == self.architecture_name:
            print(f"Already using '{name}' architecture.")
            return

        self.architecture_name = name
        self._init_architecture()
        print(f"✓ Switched to '{self.architecture.display_name}'")

    def check_health(self):
        """Verify connections to LLM and Vector DB."""
        print("\n---- Running Health Checks ----")

        try:
            print("1. Checking LLM connection...", end=" ")
            self.llm.invoke("Hello")
            print("✅ OK")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")

        try:
            print("2. Checking Vector DB connection...", end=" ")
            self.vector_store.similarity_search("test", k=1)
            print("✅ OK")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")

        print(f"3. Active Architecture: {self.architecture.display_name}")
        print("-------------------------------\n")

    def get_current_architecture(self) -> Dict[str, Any]:
        """Return info about the current architecture."""
        return self.architecture.get_info()
