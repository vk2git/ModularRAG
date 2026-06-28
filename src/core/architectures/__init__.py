"""
RAG Architecture implementations.

Each architecture is a distinct RAG pattern with its own retrieval
and generation strategy. All architectures share the same modular
components (LLM, embeddings, vector store, memory) but wire them
differently.

Available architectures:
- Naive RAG: Simple retrieve → generate
- Advanced RAG: Query rewrite + hybrid search + reranking
- Corrective RAG (CRAG): Grade docs, fallback on poor retrieval
- Self-RAG: Self-reflective generation with retries
- Agentic RAG: LLM agent with tools for multi-hop reasoning
- Adaptive RAG: Auto-routes queries to the best architecture
- Graph RAG: Knowledge graph + vector hybrid retrieval
"""
