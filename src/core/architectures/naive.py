"""
Naive RAG Architecture

The simplest RAG pattern: Query → Embed → Retrieve top-K → Generate with context.
This is the refactored version of the original rag_pipeline.py.

Best for: POCs, simple Q&A, small document collections.
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


class NaiveRAG(BaseArchitecture):
    name = "naive"
    display_name = "Naive RAG"
    description = "Simple retrieve → generate pipeline. Fast, lightweight, great for POCs."
    requires = []

    def run(self, query: str, session_id: str = "default") -> str:
        """
        Linear RAG flow:
        1. Validate input (guardrails)
        2. Retrieve relevant documents via vector similarity
        3. If docs found → RAG mode (answer from context)
        4. If no docs → General chat mode (LLM knowledge)
        5. Validate output (guardrails)
        """
        try:
            # 1. Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]

            # 2. Get chat history
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- Processing query: {sanitized_query} (Session: {session_id})")

            # 3. Retrieve with threshold filtering
            threshold = self.config.get("score_threshold", 1.5)
            k = self.config.get("top_k", 3)

            results_with_scores = self.vector_store.similarity_search_with_score(
                sanitized_query, k=k
            )

            relevant_docs = []
            for doc, score in results_with_scores:
                self._log(f"   - Doc Score: {score:.4f} | Content: {doc.page_content[:50]}...")
                if score < threshold:
                    relevant_docs.append(doc)

            # 4. Generate response
            if relevant_docs:
                self._log(f"Found {len(relevant_docs)} relevant chunks (score < {threshold}). RAG mode.")
                response = self._run_rag_mode(sanitized_query, relevant_docs, chat_history)
            else:
                self._log(f"No docs met threshold (score < {threshold}). General chat mode.")
                response = self._run_general_mode(sanitized_query, chat_history)

            # 5. Output validation
            response = self._validate_output(response)

            # 6. Save to history
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}. Check logs or configuration."

    def _run_rag_mode(self, query, docs, chat_history):
        """Answer based on retrieved context."""
        context_text = "\n\n".join([doc.page_content for doc in docs])

        template = """
        You are a specialized assistant. Answer the question based on the following context.
        
        CRITICAL INSTRUCTIONS:
        1. Answer CONCISELY. Do not ramble.
        2. If the context is NOT relevant to the question, IGNORE the context and answer from your own knowledge, but mention that you are doing so.
        3. If you don't know the answer, just say "I don't know".
        
        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": query
        })

    def _run_general_mode(self, query, chat_history):
        """Answer from LLM's own training data."""
        template = """
        You are a helpful AI assistant. Answer the following question to the best of your ability.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "chat_history": chat_history,
            "question": query
        })
