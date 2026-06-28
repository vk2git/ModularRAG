"""
Adaptive RAG Architecture

Meta-architecture that routes queries to the best RAG strategy based on
query complexity. Uses an LLM classifier to determine the optimal path.

Routing:
- Simple/factual queries → Naive RAG (fast, cheap)
- Moderate queries → Advanced RAG (rewrite + rerank)
- Complex/multi-hop queries → Corrective or Agentic RAG (thorough)

Best for: Production systems serving diverse query types where you want
to optimize cost and latency while maintaining quality.
Requires: langgraph
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict


class AdaptiveState(TypedDict):
    """State for the Adaptive RAG graph."""
    query: str
    complexity: str
    response: str


class AdaptiveRAG(BaseArchitecture):
    name = "adaptive"
    display_name = "Adaptive RAG"
    description = "Auto-routes queries to the best architecture based on complexity."
    requires = ["langgraph"]

    def run(self, query: str, session_id: str = "default") -> str:
        """Execute the Adaptive RAG pipeline."""
        try:
            # Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- [Adaptive RAG] Processing: {sanitized_query}")

            # Build and run the routing graph
            graph = self._build_graph(session_id, chat_history, history)

            initial_state = AdaptiveState(
                query=sanitized_query,
                complexity="",
                response="",
            )

            result = graph.invoke(initial_state)
            response = result.get("response", "I was unable to generate a response.")

            # Output validation
            response = self._validate_output(response)
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}"

    def _build_graph(self, session_id, chat_history, history):
        """Build the Adaptive RAG routing graph."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AdaptiveState)

        # Define nodes
        workflow.add_node("classify", lambda s: self._classify_query(s))
        workflow.add_node("run_naive", lambda s: self._run_naive(s, session_id, chat_history, history))
        workflow.add_node("run_advanced", lambda s: self._run_advanced(s, session_id, chat_history, history))
        workflow.add_node("run_corrective", lambda s: self._run_corrective(s, session_id))
        workflow.add_node("run_agentic", lambda s: self._run_agentic(s, session_id))

        # Define flow
        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify",
            self._route_by_complexity,
            {
                "naive": "run_naive",
                "advanced": "run_advanced",
                "corrective": "run_corrective",
                "agentic": "run_agentic",
            }
        )

        # All routes end after execution
        workflow.add_edge("run_naive", END)
        workflow.add_edge("run_advanced", END)
        workflow.add_edge("run_corrective", END)
        workflow.add_edge("run_agentic", END)

        return workflow.compile()

    def _classify_query(self, state: AdaptiveState) -> dict:
        """Classify query complexity using LLM."""
        self._log("---- Classifying query complexity...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a query complexity classifier for a RAG system. "
                "Classify the query into ONE of these categories:\n\n"
                "- 'simple': Direct factual questions, lookups, definitions "
                "(e.g., 'What is X?', 'When did Y happen?')\n"
                "- 'moderate': Questions requiring synthesis from multiple sections "
                "or some context understanding (e.g., 'How does X compare to Y?')\n"
                "- 'complex': Multi-hop reasoning, analytical questions, or questions "
                "requiring multiple retrieval passes (e.g., 'What are the implications of X on Y given Z?')\n"
                "- 'exploratory': Open-ended research questions that may need web search "
                "(e.g., 'What are the latest trends in X?')\n\n"
                "Respond with ONLY one word: simple, moderate, complex, or exploratory."
            )),
            ("human", "Classify this query: {query}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            complexity = chain.invoke({"query": state["query"]}).strip().lower()
            # Normalize to valid values
            if complexity not in ("simple", "moderate", "complex", "exploratory"):
                complexity = "moderate"

            self._log(f"---- Query classified as: {complexity}")
            return {"complexity": complexity}
        except Exception as e:
            self._log(f"⚠️  Classification failed: {e}. Defaulting to moderate.")
            return {"complexity": "moderate"}

    def _route_by_complexity(self, state: AdaptiveState) -> str:
        """Route to the appropriate architecture based on complexity."""
        routing = self.config.get("routing", {})

        routes = {
            "simple": routing.get("simple", "naive"),
            "moderate": routing.get("moderate", "advanced"),
            "complex": routing.get("complex", "corrective"),
            "exploratory": routing.get("exploratory", "agentic"),
        }

        target = routes.get(state["complexity"], "advanced")
        self._log(f"---- Routing to: {target}")
        return target

    def _run_naive(self, state: AdaptiveState, session_id, chat_history, history) -> dict:
        """Run Naive RAG for simple queries."""
        self._log("---- Dispatching to Naive RAG...")
        from src.core.architectures.naive import NaiveRAG

        arch = NaiveRAG(
            components=self.components,
            config=self.config,
            verbose=self.verbose
        )

        # Direct run to avoid double guardrails/history
        k = self.config.get("top_k", 3)
        results = self.vector_store.similarity_search_with_score(state["query"], k=k)
        threshold = self.config.get("score_threshold", 1.5)
        docs = [doc for doc, score in results if score < threshold]

        if docs:
            response = arch._run_rag_mode(state["query"], docs, chat_history)
        else:
            response = arch._run_general_mode(state["query"], chat_history)

        return {"response": response}

    def _run_advanced(self, state: AdaptiveState, session_id, chat_history, history) -> dict:
        """Run Advanced RAG for moderate queries."""
        self._log("---- Dispatching to Advanced RAG...")
        from src.core.architectures.advanced import AdvancedRAG

        arch = AdvancedRAG(
            components=self.components,
            config=self.config,
            verbose=self.verbose
        )

        # Use the advanced pipeline's core logic
        rewritten = arch._rewrite_query(state["query"], chat_history)
        from src.core.components.retriever import RetrieverFactory
        rf = RetrieverFactory(self.vector_store, self.embeddings)

        try:
            retriever = rf.create_retriever(strategy="reranked", k=5)
            docs = retriever.invoke(rewritten)
        except Exception:
            retriever = rf.create_retriever(strategy="basic", k=5)
            docs = retriever.invoke(rewritten)

        if docs:
            response = arch._generate_with_context(state["query"], docs, chat_history)
        else:
            response = arch._generate_general(state["query"], chat_history)

        return {"response": response}

    def _run_corrective(self, state: AdaptiveState, session_id) -> dict:
        """Run Corrective RAG for complex queries."""
        self._log("---- Dispatching to Corrective RAG...")
        from src.core.architectures.corrective import CorrectiveRAG

        arch = CorrectiveRAG(
            components=self.components,
            config=self.config,
            verbose=self.verbose
        )
        response = arch.run(state["query"], session_id)
        return {"response": response}

    def _run_agentic(self, state: AdaptiveState, session_id) -> dict:
        """Run Agentic RAG for exploratory queries."""
        self._log("---- Dispatching to Agentic RAG...")
        from src.core.architectures.agentic import AgenticRAG

        arch = AgenticRAG(
            components=self.components,
            config=self.config,
            verbose=self.verbose
        )
        response = arch.run(state["query"], session_id)
        return {"response": response}
