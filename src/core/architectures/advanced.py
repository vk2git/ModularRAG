"""
Advanced RAG Architecture

Enhances Naive RAG with three key improvements:
1. Query Rewriting: LLM rewrites the user query for better retrieval
2. Hybrid Search: Combines vector similarity + keyword matching (BM25)
3. Reranking: Cross-encoder reranks retrieved docs for relevance

Flow: Query → Rewrite → Hybrid Retrieve → Rerank → Generate

Best for: Production systems needing high retrieval quality.
"""

from src.core.architectures.base import BaseArchitecture
from src.core.components.retriever import RetrieverFactory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


class AdvancedRAG(BaseArchitecture):
    name = "advanced"
    display_name = "Advanced RAG"
    description = "Query rewrite + hybrid search + reranking for production-quality retrieval."
    requires = []

    def run(self, query: str, session_id: str = "default") -> str:
        """
        Advanced RAG flow:
        1. Validate input
        2. Rewrite query for better retrieval
        3. Retrieve via configured strategy (hybrid/reranked)
        4. Generate response from reranked context
        5. Validate output
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

            self._log(f"---- [Advanced RAG] Processing: {sanitized_query}")

            # 3. Query rewriting
            rewrite_enabled = self.config.get("query_rewrite", True)
            if rewrite_enabled:
                rewritten_query = self._rewrite_query(sanitized_query, chat_history)
                self._log(f"---- Rewritten query: {rewritten_query}")
            else:
                rewritten_query = sanitized_query

            # 4. Retrieve with configured strategy
            retriever_strategy = self.config.get("retriever_strategy", "reranked")
            k = self.config.get("top_k", 5)

            retriever_factory = RetrieverFactory(
                vector_store=self.vector_store,
                embeddings=self.embeddings
            )

            try:
                retriever = retriever_factory.create_retriever(
                    strategy=retriever_strategy, k=k
                )
                docs = retriever.invoke(rewritten_query)
            except (ImportError, Exception) as e:
                self._log(f"⚠️  {retriever_strategy} retriever failed ({e}), falling back to basic")
                retriever = retriever_factory.create_retriever(strategy="basic", k=k)
                docs = retriever.invoke(rewritten_query)

            self._log(f"---- Retrieved {len(docs)} documents")

            # 5. Generate response
            if docs:
                response = self._generate_with_context(
                    sanitized_query, docs, chat_history
                )
            else:
                self._log("No documents retrieved. Using general chat.")
                response = self._generate_general(sanitized_query, chat_history)

            # 6. Output validation
            response = self._validate_output(response)

            # 7. Save to history
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}. Check logs or configuration."

    def _rewrite_query(self, query: str, chat_history) -> str:
        """Use LLM to rewrite the query for better retrieval."""
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a query rewriter. Your job is to reformulate the user's question "
                "to make it more specific and better suited for document retrieval. "
                "Consider the conversation history for context. "
                "Output ONLY the rewritten query, nothing else."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Rewrite this query for better document retrieval: {query}")
        ])

        chain = rewrite_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "chat_history": chat_history[-4:] if chat_history else []
            })
            return result.strip()
        except Exception as e:
            self._log(f"⚠️  Query rewrite failed: {e}. Using original query.")
            return query

    def _generate_with_context(self, query, docs, chat_history):
        """Generate answer from retrieved and reranked documents."""
        context_text = "\n\n".join([doc.page_content for doc in docs])

        template = """
        You are a knowledgeable assistant. Answer the question using the provided context.
        
        INSTRUCTIONS:
        1. Be CONCISE and ACCURATE.
        2. Cite specific parts of the context when possible.
        3. If the context doesn't contain the answer, say so clearly and answer from your knowledge.
        4. If you're unsure, say "I don't know".
        
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

    def _generate_general(self, query, chat_history):
        """Fallback: answer from LLM's own knowledge."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the question to the best of your ability."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"chat_history": chat_history, "question": query})
