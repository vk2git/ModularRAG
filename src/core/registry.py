"""
Architecture Registry — discovers and manages all available RAG architectures.

The registry auto-discovers architecture classes from src/core/architectures/
and provides a unified interface to list, get, and check their status.
"""

from typing import Dict, List, Optional, Type
from src.core.architectures.base import BaseArchitecture


# All available architectures — import once, register here
_ARCHITECTURE_CLASSES: Dict[str, Type[BaseArchitecture]] = {}


def _discover_architectures():
    """Import and register all built-in architectures."""
    global _ARCHITECTURE_CLASSES

    if _ARCHITECTURE_CLASSES:
        return  # Already discovered

    from src.core.architectures.naive import NaiveRAG
    from src.core.architectures.advanced import AdvancedRAG
    from src.core.architectures.corrective import CorrectiveRAG
    from src.core.architectures.self_rag import SelfRAG
    from src.core.architectures.agentic import AgenticRAG
    from src.core.architectures.adaptive import AdaptiveRAG
    from src.core.architectures.graph_rag import GraphRAG

    for cls in [NaiveRAG, AdvancedRAG, CorrectiveRAG, SelfRAG, AgenticRAG, AdaptiveRAG, GraphRAG]:
        _ARCHITECTURE_CLASSES[cls.name] = cls


class ArchitectureRegistry:
    """
    Central registry for all RAG architectures.
    
    Usage:
        registry = ArchitectureRegistry()
        registry.list_architectures()  # List all
        registry.get_architecture_class("naive")  # Get class by name
    """

    def __init__(self):
        _discover_architectures()

    def list_architectures(self) -> List[Dict[str, str]]:
        """
        List all registered architectures with their status.

        Returns:
            List of dicts with name, display_name, description, status
        """
        architectures = []
        for name, cls in _ARCHITECTURE_CLASSES.items():
            req_check = self._check_requirements(cls)
            opt_check = self._check_optional_deps(cls)

            if not req_check["available"]:
                status = f"❌ Missing: {', '.join(req_check['missing'])}"
            elif opt_check["missing"]:
                status = f"✅ Ready (optional: {', '.join(opt_check['missing'])} not installed)"
            else:
                status = "✅ Ready"

            architectures.append({
                "name": cls.name,
                "display_name": cls.display_name,
                "description": cls.description,
                "requires": cls.requires,
                "optional_deps": cls.optional_deps,
                "status": status,
                "available": req_check["available"],
            })
        return architectures

    def get_architecture_class(self, name: str) -> Type[BaseArchitecture]:
        """
        Get an architecture class by name.

        Args:
            name: Architecture name (e.g., "naive", "advanced")

        Returns:
            The architecture class (not instantiated)

        Raises:
            ValueError: If architecture not found
        """
        if name not in _ARCHITECTURE_CLASSES:
            available = ", ".join(_ARCHITECTURE_CLASSES.keys())
            raise ValueError(
                f"Unknown architecture: '{name}'. Available: {available}"
            )
        return _ARCHITECTURE_CLASSES[name]

    def get_names(self) -> List[str]:
        """Return list of all architecture names."""
        return list(_ARCHITECTURE_CLASSES.keys())

    def is_available(self, name: str) -> bool:
        """Check if an architecture is available (all requirements met)."""
        try:
            cls = self.get_architecture_class(name)
            return self._check_requirements(cls)["available"]
        except ValueError:
            return False

    def _check_requirements(self, cls: Type[BaseArchitecture]) -> Dict:
        """Check if hard dependencies for an architecture are installed."""
        missing = []
        for req in cls.requires:
            try:
                __import__(req.replace("-", "_"))
            except ImportError:
                missing.append(req)
        return {"available": len(missing) == 0, "missing": missing}

    def _check_optional_deps(self, cls: Type[BaseArchitecture]) -> Dict:
        """Check if optional (soft) dependencies are installed."""
        missing = []
        for dep in cls.optional_deps:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing.append(dep)
        return {"available": len(missing) == 0, "missing": missing}
