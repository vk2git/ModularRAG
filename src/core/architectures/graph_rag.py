"""
Graph RAG Architecture

Combines knowledge graph traversal with vector search for
relationship-aware retrieval. Entities and their relationships
are extracted from documents and stored in a knowledge graph.

Two retrieval paths run in parallel:
1. Vector similarity search (standard)
2. Knowledge graph traversal (relationship-aware)

Results are merged for richer, more connected context.

Best for: Enterprise data with complex relationships (compliance, finance,
organizational data) where "connecting the dots" matters.

Requires: neo4j, langchain-neo4j (install with: uv pip install 'modular-rag[graph]')
"""

from src.core.architectures.base import BaseArchitecture
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List


class GraphRAG(BaseArchitecture):
    name = "graph_rag"
    display_name = "Graph RAG"
    description = "Knowledge graph + vector hybrid retrieval for relationship-aware answers."
    requires = ["neo4j"]

    def __init__(self, components, config, verbose=False):
        super().__init__(components, config, verbose)
        self.graph_store = None
        self._init_graph_store()

    def _init_graph_store(self):
        """Initialize Neo4j graph store if available."""
        graph_config = self.config.get("graph", {})
        if not graph_config.get("enabled", True):
            self._log("Graph store disabled in config")
            return

        try:
            from langchain_neo4j import Neo4jGraph
            import os

            uri = graph_config.get("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
            username = graph_config.get("username", os.getenv("NEO4J_USERNAME", "neo4j"))
            password = graph_config.get("password", os.getenv("NEO4J_PASSWORD", ""))

            if not password:
                self._log("⚠️  NEO4J_PASSWORD not set. Graph features disabled.")
                return

            self.graph_store = Neo4jGraph(
                url=uri,
                username=username,
                password=password,
            )
            self._log(f"---- Connected to Neo4j at {uri}")

        except ImportError:
            self._log("⚠️  neo4j/langchain-neo4j not installed. Install with: uv pip install 'modular-rag[graph]'")
        except Exception as e:
            self._log(f"⚠️  Failed to connect to Neo4j: {e}")

    def run(self, query: str, session_id: str = "default") -> str:
        """Execute the Graph RAG pipeline."""
        try:
            # Input validation
            validation = self._validate_input(query)
            if not validation["valid"]:
                print(f"⚠️  Input validation failed: {validation['reason']}")
                return self.guardrails.get_safe_response(validation)

            sanitized_query = validation["sanitized_input"]
            history, chat_history = self._get_chat_history(session_id)

            self._log(f"---- [Graph RAG] Processing: {sanitized_query}")

            # 1. Vector retrieval (always available)
            k = self.config.get("top_k", 5)
            vector_docs = self.vector_store.similarity_search(sanitized_query, k=k)
            self._log(f"---- Vector search: {len(vector_docs)} documents")

            # 2. Graph retrieval (if graph store is available)
            graph_context = ""
            if self.graph_store:
                graph_context = self._query_knowledge_graph(sanitized_query)
                self._log(f"---- Graph search: {len(graph_context)} chars of context")

            # 3. Merge and generate
            response = self._generate_with_hybrid_context(
                sanitized_query, vector_docs, graph_context, chat_history
            )

            # Output validation
            response = self._validate_output(response)
            self._save_to_history(history, query, response)

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing request: {e}"

    def _query_knowledge_graph(self, query: str) -> str:
        """Query the knowledge graph for related entities and relationships."""
        if not self.graph_store:
            return ""

        try:
            # Use LLM to generate a Cypher query from the natural language question
            cypher_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a Neo4j Cypher expert. Convert the user's question into a Cypher query.\n"
                    "The graph schema is:\n{schema}\n\n"
                    "Rules:\n"
                    "- Return relevant nodes and relationships\n"
                    "- Limit results to 10\n"
                    "- If the question doesn't map to the schema, return: MATCH (n) RETURN n LIMIT 0\n"
                    "- Output ONLY the Cypher query, no explanation"
                )),
                ("human", "Question: {question}")
            ])

            chain = cypher_prompt | self.llm | StrOutputParser()

            schema = self.graph_store.get_schema
            cypher_query = chain.invoke({
                "schema": str(schema)[:2000],
                "question": query
            }).strip()

            # Remove markdown code blocks if present
            if cypher_query.startswith("```"):
                lines = cypher_query.split("\n")
                cypher_query = "\n".join(lines[1:-1])

            self._log(f"---- Cypher query: {cypher_query}")

            # Execute the query
            results = self.graph_store.query(cypher_query)

            if results:
                return "\n".join([str(r) for r in results[:10]])
            return ""

        except Exception as e:
            self._log(f"⚠️  Graph query failed: {e}")
            return ""

    def _generate_with_hybrid_context(self, query, vector_docs, graph_context, chat_history):
        """Generate answer from both vector and graph contexts."""
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])

        # Build combined context
        context_parts = []
        if vector_context:
            context_parts.append(f"## Document Context\n{vector_context}")
        if graph_context:
            context_parts.append(f"## Knowledge Graph Context (Entity Relationships)\n{graph_context}")

        combined_context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        template = """
        You are a knowledgeable assistant with access to both document search and a knowledge graph.
        Answer the question using the provided context. The knowledge graph context shows 
        entity relationships that may help connect information across documents.
        
        INSTRUCTIONS:
        1. Be CONCISE and ACCURATE.
        2. Use both document and graph context when available.
        3. Highlight any relationships or connections you find in the graph data.
        4. If context is insufficient, say so honestly.
        
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
            "context": combined_context,
            "chat_history": chat_history,
            "question": query
        })

    def extract_and_store_entities(self, documents: List[Document]):
        """
        Extract entities and relationships from documents and store in the knowledge graph.
        Called during ingestion to populate the graph.
        """
        if not self.graph_store:
            self._log("⚠️  Graph store not available. Skipping entity extraction.")
            return

        self._log("---- Extracting entities from documents...")

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Extract entities and relationships from the text. "
                "Format as Cypher CREATE statements.\n"
                "Example:\n"
                "CREATE (a:Person {{name: 'John'}})\n"
                "CREATE (b:Company {{name: 'Acme'}})\n"
                "CREATE (a)-[:WORKS_AT]->(b)\n\n"
                "Output ONLY Cypher CREATE statements, one per line."
            )),
            ("human", "Text: {text}")
        ])

        chain = extraction_prompt | self.llm | StrOutputParser()

        for doc in documents[:50]:  # Limit for safety
            try:
                cypher = chain.invoke({"text": doc.page_content[:2000]}).strip()

                if cypher and "CREATE" in cypher.upper():
                    # Remove markdown code blocks
                    if cypher.startswith("```"):
                        lines = cypher.split("\n")
                        cypher = "\n".join(lines[1:-1])

                    self.graph_store.query(cypher)
                    self._log(f"   ✓ Extracted entities from: {doc.metadata.get('source', 'unknown')[:50]}")
            except Exception as e:
                self._log(f"   ✗ Entity extraction failed: {e}")
