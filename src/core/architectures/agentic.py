"""
Agentic RAG Architecture

An LLM agent with tools that can iteratively plan, retrieve, and reason.
Uses LangGraph to give the LLM control over when and how to retrieve.

The agent has access to:
- retrieve_docs: Search the local vector store
- web_search: Search the web (if enabled)

The agent decides:
- Whether to retrieve at all
- How many retrieval passes to make
- When it has enough information to answer

Best for: Complex, multi-hop questions requiring iterative reasoning.
Requires: langgraph
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import List, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgenticState(TypedDict):
    """State for the Agentic RAG graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class AgenticRAG(BaseArchitecture):
    name = "agentic"
    display_name = "Agentic RAG"
    description = "LLM agent with tools for iterative, multi-hop reasoning and retrieval."
    requires = ["langgraph"]

    def run(self, query: str, session_id: str = "default") -> str:
        """Execute the Agentic RAG pipeline."""
        try:
            # Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- [Agentic RAG] Processing: {sanitized_query}")

            # Build tools and graph
            tools = self._build_tools()
            graph = self._build_graph(tools)

            # Prepare messages
            system_msg = SystemMessage(content=(
                "You are a helpful research assistant with access to document search tools. "
                "Use the tools to find information before answering. "
                "You may call tools multiple times if needed. "
                "Always base your final answer on the retrieved information. "
                "If you can't find relevant information, say so honestly."
            ))

            messages = [system_msg] + list(chat_history[-4:]) + [HumanMessage(content=sanitized_query)]

            initial_state = AgenticState(messages=messages)

            # Run the agent
            max_steps = self.config.get("max_agent_steps", 10)
            result = graph.invoke(
                initial_state,
                config={"recursion_limit": max_steps}
            )

            # Extract final response
            final_messages = result.get("messages", [])
            if final_messages:
                response = final_messages[-1].content
            else:
                response = "I was unable to generate a response."

            # Output validation
            response = self._validate_output(response)
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}"

    def _build_tools(self):
        """Build the tools available to the agent."""
        tools_list = []

        # Document retrieval tool
        vector_store = self.vector_store
        k = self.config.get("top_k", 5)

        @tool
        def retrieve_documents(query: str) -> str:
            """Search the local document knowledge base for relevant information. Use this for questions about ingested documents."""
            docs = vector_store.similarity_search(query, k=k)
            if not docs:
                return "No relevant documents found in the knowledge base."
            return "\n\n---\n\n".join([
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ])

        tools_list.append(retrieve_documents)

        # Web search tool (if enabled)
        web_search_enabled = self.config.get("web_search", True)
        if web_search_enabled:
            try:
                from src.core.components.web_search import WebSearchTool
                web_tool = WebSearchTool()
                langchain_tool = web_tool.as_langchain_tool()
                if langchain_tool:
                    tools_list.append(langchain_tool)
                    self._log("---- Web search tool enabled for agent")
            except Exception as e:
                self._log(f"⚠️  Web search not available: {e}")

        return tools_list

    def _build_graph(self, tools):
        """Build the agent graph using LangGraph's prebuilt react agent."""
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(self.llm, tools)
