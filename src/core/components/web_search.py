"""
Web search tool for RAG architectures that need external search fallback.

Used by: Corrective RAG, Agentic RAG, Adaptive RAG

Providers:
- tavily (default): Tavily Search API (requires TAVILY_API_KEY)
- duckduckgo: DuckDuckGo search (free, no API key)

Enabled by default. Users can disable via config:
  web_search:
    enabled: false
"""

from src.utils.config_loader import load_config
from langchain_core.documents import Document
from typing import List
import os


class WebSearchTool:
    """Provides web search functionality for RAG fallback."""

    def __init__(self):
        self.config = load_config()
        self.search_config = self.config.get("web_search", {})
        self.enabled = self.search_config.get("enabled", True)
        self.provider = self.search_config.get("provider", "duckduckgo").lower()
        self.max_results = self.search_config.get("max_results", 3)
        self._tool = None

    def is_available(self) -> bool:
        """Check if web search is enabled and the provider is installed."""
        if not self.enabled:
            return False
        try:
            self._get_tool()
            return True
        except (ImportError, ValueError):
            return False

    def _get_tool(self):
        """Lazy-load the search tool."""
        if self._tool is not None:
            return self._tool

        if self.provider == "tavily":
            self._tool = self._create_tavily()
        elif self.provider == "duckduckgo":
            self._tool = self._create_duckduckgo()
        else:
            raise ValueError(f"Unsupported web search provider: {self.provider}")

        return self._tool

    def _create_tavily(self):
        """Create Tavily search tool."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except ImportError:
            raise ImportError(
                "Tavily search requires 'tavily-python'. "
                "Install with: uv pip install 'modular-rag[search]'"
            )

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not found in environment variables. "
                "Get a free key at https://tavily.com"
            )

        return TavilySearchResults(max_results=self.max_results)

    def _create_duckduckgo(self):
        """Create DuckDuckGo search tool (free, no API key)."""
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
        except ImportError:
            raise ImportError(
                "DuckDuckGo search requires 'duckduckgo-search'. "
                "Install with: uv pip install 'modular-rag[search]'"
            )

        return DuckDuckGoSearchResults(max_results=self.max_results)

    def search(self, query: str) -> List[Document]:
        """
        Perform a web search and return results as LangChain Documents.

        Args:
            query: The search query

        Returns:
            List of Documents with web search results
        """
        if not self.enabled:
            return []

        try:
            tool = self._get_tool()
            results = tool.invoke(query)

            # Convert results to Documents
            if isinstance(results, str):
                return [Document(page_content=results, metadata={"source": "web_search"})]
            elif isinstance(results, list):
                docs = []
                for result in results:
                    if isinstance(result, dict):
                        content = result.get("content", result.get("snippet", str(result)))
                        url = result.get("url", "web_search")
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": url, "type": "web_search"}
                        ))
                    else:
                        docs.append(Document(
                            page_content=str(result),
                            metadata={"source": "web_search"}
                        ))
                return docs
            else:
                return [Document(page_content=str(results), metadata={"source": "web_search"})]

        except Exception as e:
            print(f"⚠️  Web search failed: {e}")
            return []

    def as_langchain_tool(self):
        """
        Return the underlying LangChain tool for use in agent architectures.
        """
        if not self.enabled:
            return None
        try:
            return self._get_tool()
        except (ImportError, ValueError) as e:
            print(f"⚠️  Web search tool not available: {e}")
            return None
