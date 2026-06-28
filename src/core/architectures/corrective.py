"""
Corrective RAG (CRAG) Architecture

Implements a retrieval quality gate using LangGraph:
1. Retrieve documents
2. Grade each document for relevance (LLM-as-judge)
3. If relevant docs found → Generate answer
4. If not → Rewrite query and re-retrieve (or web search fallback)

Flow:
  [Retrieve] → [Grade Docs] → relevant? → [Generate]
                             → not relevant? → [Rewrite/Web Search] → [Generate]

Best for: High-accuracy systems where hallucination is unacceptable.
Requires: langgraph
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, TypedDict, Annotated
import operator


class CRAGState(TypedDict):
    """State for the Corrective RAG graph."""
    query: str
    documents: List[Document]
    generation: str
    web_search_needed: bool
    retries: int
    max_retries: int


class CorrectiveRAG(BaseArchitecture):
    name = "corrective"
    display_name = "Corrective RAG (CRAG)"
    description = "Grades retrieved docs, rewrites query or falls back to web search if retrieval is poor."
    requires = ["langgraph"]

    def run(self, query: str, session_id: str = "default") -> str:
        """Execute the Corrective RAG pipeline via LangGraph."""
        try:
            # Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- [Corrective RAG] Processing: {sanitized_query}")

            # Build and run the graph
            graph = self._build_graph(chat_history)

            max_retries = self.config.get("max_retries", 2)
            initial_state = CRAGState(
                query=sanitized_query,
                documents=[],
                generation="",
                web_search_needed=False,
                retries=0,
                max_retries=max_retries,
            )

            result = graph.invoke(initial_state)
            response = result.get("generation", "I was unable to generate a response.")

            # Output validation
            response = self._validate_output(response)
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}"

    def _build_graph(self, chat_history):
        """Build the CRAG LangGraph workflow."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(CRAGState)

        # Define nodes
        workflow.add_node("retrieve", lambda state: self._retrieve(state))
        workflow.add_node("grade_documents", lambda state: self._grade_documents(state))
        workflow.add_node("generate", lambda state: self._generate(state, chat_history))
        workflow.add_node("rewrite_query", lambda state: self._rewrite_query(state))
        workflow.add_node("web_search", lambda state: self._web_search(state))

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # After grading: if relevant docs → generate, else → rewrite or web search
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "web_search": "web_search",
            }
        )

        # After rewrite → retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        # After web search → generate
        workflow.add_edge("web_search", "generate")

        # Generate → END
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve(self, state: CRAGState) -> dict:
        """Retrieve documents from vector store."""
        k = self.config.get("top_k", 5)
        self._log(f"---- Retrieving documents for: {state['query']}")

        docs = self.vector_store.similarity_search(state["query"], k=k)
        self._log(f"---- Retrieved {len(docs)} documents")

        return {"documents": docs}

    def _grade_documents(self, state: CRAGState) -> dict:
        """Grade retrieved documents for relevance using LLM-as-judge."""
        self._log("---- Grading documents for relevance...")

        grading_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a document relevance grader. Given a user question and a document, "
                "determine if the document is relevant to answering the question.\n"
                "Respond with ONLY 'yes' or 'no'."
            )),
            ("human", "Question: {question}\n\nDocument: {document}\n\nIs this document relevant?")
        ])

        chain = grading_prompt | self.llm | StrOutputParser()

        relevant_docs = []
        for doc in state["documents"]:
            try:
                grade = chain.invoke({
                    "question": state["query"],
                    "document": doc.page_content[:1000]  # Limit for grading
                }).strip().lower()

                if "yes" in grade:
                    relevant_docs.append(doc)
                    self._log(f"   ✓ Relevant: {doc.page_content[:50]}...")
                else:
                    self._log(f"   ✗ Not relevant: {doc.page_content[:50]}...")
            except Exception as e:
                self._log(f"   ⚠️  Grading failed for doc: {e}")
                relevant_docs.append(doc)  # Keep on error

        web_search_needed = len(relevant_docs) == 0
        self._log(f"---- {len(relevant_docs)}/{len(state['documents'])} documents deemed relevant")

        return {
            "documents": relevant_docs,
            "web_search_needed": web_search_needed,
        }

    def _decide_after_grading(self, state: CRAGState) -> str:
        """Decide next step after document grading."""
        if state["documents"]:
            return "generate"

        # No relevant docs — check if we should retry or web search
        if state["retries"] < state["max_retries"]:
            return "rewrite"

        # Check if web search is available
        web_search_enabled = self.config.get("web_search", True)
        if web_search_enabled:
            return "web_search"

        # Exhausted options, generate with what we have
        return "generate"

    def _rewrite_query(self, state: CRAGState) -> dict:
        """Rewrite the query for better retrieval."""
        self._log("---- Rewriting query for better retrieval...")

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a query rewriter. The previous search didn't find relevant documents. "
                "Reformulate the question to improve search results. "
                "Output ONLY the rewritten query."
            )),
            ("human", "Original question: {query}\n\nRewrite this for better document search:")
        ])

        chain = rewrite_prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({"query": state["query"]}).strip()
        self._log(f"---- Rewritten query: {rewritten}")

        return {
            "query": rewritten,
            "retries": state["retries"] + 1,
        }

    def _web_search(self, state: CRAGState) -> dict:
        """Fallback to web search when local docs aren't relevant."""
        self._log("---- Falling back to web search...")

        from src.core.components.web_search import WebSearchTool
        search_tool = WebSearchTool()

        if search_tool.is_available():
            web_docs = search_tool.search(state["query"])
            self._log(f"---- Web search returned {len(web_docs)} results")
            return {"documents": web_docs}
        else:
            self._log("⚠️  Web search not available. Generating without context.")
            return {"documents": []}

    def _generate(self, state: CRAGState, chat_history) -> dict:
        """Generate answer from documents."""
        self._log("---- Generating response...")

        if state["documents"]:
            context = "\n\n".join([doc.page_content for doc in state["documents"]])

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful assistant. Answer the question using the provided context.\n"
                    "Be concise and accurate. If the context doesn't help, say so.\n\n"
                    "Context:\n{context}"
                )),
                ("human", "{question}")
            ])

            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": state["query"]
            })
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer from your own knowledge. Be honest if you're unsure."),
                ("human", "{question}")
            ])

            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": state["query"]})

        return {"generation": response}
