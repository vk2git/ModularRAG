"""
Self-RAG Architecture

Self-reflective RAG that critiques its own output using LangGraph:
1. Retrieve → Generate an answer
2. Check: Is the answer grounded in the retrieved documents?
3. Check: Does the answer actually address the question?
4. If either check fails → Rewrite query and loop back (max N iterations)

Flow:
  [Retrieve] → [Generate] → [Check Hallucination] → grounded?
       ↑                         → yes → [Check Answer] → useful? → Return
       └──── [Rewrite Query] ←── no ──────────────────── no ──────┘

Best for: Systems where hallucination control is critical.
Requires: langgraph
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, TypedDict


class SelfRAGState(TypedDict):
    """State for the Self-RAG graph."""
    query: str
    documents: List[Document]
    generation: str
    is_grounded: bool
    is_useful: bool
    iteration: int
    max_iterations: int


class SelfRAG(BaseArchitecture):
    name = "self_rag"
    display_name = "Self-RAG"
    description = "Self-reflective generation with hallucination checks and automatic retries."
    requires = ["langgraph"]

    def run(self, query: str, session_id: str = "default") -> str:
        """Execute the Self-RAG pipeline via LangGraph."""
        try:
            # Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- [Self-RAG] Processing: {sanitized_query}")

            # Build and run the graph
            graph = self._build_graph(chat_history)

            max_iterations = self.config.get("max_iterations", 3)
            initial_state = SelfRAGState(
                query=sanitized_query,
                documents=[],
                generation="",
                is_grounded=False,
                is_useful=False,
                iteration=0,
                max_iterations=max_iterations,
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
        """Build the Self-RAG LangGraph workflow."""
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(SelfRAGState)

        # Define nodes
        workflow.add_node("retrieve", lambda s: self._retrieve(s))
        workflow.add_node("generate", lambda s: self._generate(s, chat_history))
        workflow.add_node("check_hallucination", lambda s: self._check_hallucination(s))
        workflow.add_node("check_answer", lambda s: self._check_answer(s))
        workflow.add_node("rewrite_query", lambda s: self._rewrite_query(s))

        # Define flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "check_hallucination")

        # After hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            self._decide_after_hallucination_check,
            {
                "check_answer": "check_answer",
                "rewrite": "rewrite_query",
                "end": END,
            }
        )

        # After answer check
        workflow.add_conditional_edges(
            "check_answer",
            self._decide_after_answer_check,
            {
                "end": END,
                "rewrite": "rewrite_query",
            }
        )

        # After rewrite → retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        return workflow.compile()

    def _retrieve(self, state: SelfRAGState) -> dict:
        """Retrieve documents."""
        k = self.config.get("top_k", 5)
        self._log(f"---- [Iteration {state['iteration']}] Retrieving for: {state['query']}")

        docs = self.vector_store.similarity_search(state["query"], k=k)
        self._log(f"---- Retrieved {len(docs)} documents")

        return {"documents": docs}

    def _generate(self, state: SelfRAGState, chat_history) -> dict:
        """Generate answer from retrieved documents."""
        self._log("---- Generating response...")

        if state["documents"]:
            context = "\n\n".join([doc.page_content for doc in state["documents"]])
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Answer the question using ONLY the provided context. "
                    "If the context doesn't contain the answer, say 'I don't have enough information.'\n\n"
                    "Context:\n{context}"
                )),
                ("human", "{question}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": state["query"]})
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer honestly."),
                ("human", "{question}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": state["query"]})

        return {"generation": response}

    def _check_hallucination(self, state: SelfRAGState) -> dict:
        """Check if the generation is grounded in the retrieved documents."""
        self._log("---- Checking for hallucinations...")

        if not state["documents"]:
            self._log("   No documents to check against - skipping hallucination check")
            return {"is_grounded": True}

        context = "\n\n".join([doc.page_content for doc in state["documents"]])

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a hallucination grader. Determine if the assistant's answer "
                "is grounded in / supported by the provided documents.\n"
                "Respond with ONLY 'yes' or 'no'."
            )),
            ("human", (
                "Documents:\n{documents}\n\n"
                "Answer:\n{generation}\n\n"
                "Is the answer grounded in the documents?"
            ))
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "documents": context[:3000],
                "generation": state["generation"]
            }).strip().lower()

            is_grounded = "yes" in result
            self._log(f"   Grounded: {'✓ Yes' if is_grounded else '✗ No'}")
            return {"is_grounded": is_grounded}
        except Exception as e:
            self._log(f"   ⚠️  Hallucination check failed: {e}")
            return {"is_grounded": True}  # Proceed on error

    def _check_answer(self, state: SelfRAGState) -> dict:
        """Check if the generation actually answers the question."""
        self._log("---- Checking answer quality...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer quality grader. Determine if the answer "
                "actually addresses and resolves the user's question.\n"
                "Respond with ONLY 'yes' or 'no'."
            )),
            ("human", (
                "Question: {question}\n\n"
                "Answer: {generation}\n\n"
                "Does the answer resolve the question?"
            ))
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "question": state["query"],
                "generation": state["generation"]
            }).strip().lower()

            is_useful = "yes" in result
            self._log(f"   Useful: {'✓ Yes' if is_useful else '✗ No'}")
            return {"is_useful": is_useful}
        except Exception as e:
            self._log(f"   ⚠️  Answer check failed: {e}")
            return {"is_useful": True}  # Proceed on error

    def _decide_after_hallucination_check(self, state: SelfRAGState) -> str:
        """Decide what to do after hallucination check."""
        if state["is_grounded"]:
            return "check_answer"

        if state["iteration"] < state["max_iterations"]:
            self._log("   → Answer not grounded. Rewriting query...")
            return "rewrite"

        self._log("   → Max iterations reached. Returning best effort.")
        return "end"

    def _decide_after_answer_check(self, state: SelfRAGState) -> str:
        """Decide what to do after answer quality check."""
        if state["is_useful"]:
            return "end"

        if state["iteration"] < state["max_iterations"]:
            self._log("   → Answer not useful. Rewriting query...")
            return "rewrite"

        self._log("   → Max iterations reached. Returning best effort.")
        return "end"

    def _rewrite_query(self, state: SelfRAGState) -> dict:
        """Rewrite the query for better retrieval."""
        self._log("---- Rewriting query...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a query rewriter. The previous answer was not satisfactory. "
                "Reformulate the question to find better information. "
                "Output ONLY the rewritten query."
            )),
            ("human", "Original: {query}\nRewrite for better results:")
        ])

        chain = prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({"query": state["query"]}).strip()
        self._log(f"---- Rewritten: {rewritten}")

        return {
            "query": rewritten,
            "iteration": state["iteration"] + 1,
        }
