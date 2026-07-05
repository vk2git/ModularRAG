# Graph RAG

**Relationship-aware retrieval.** Combines knowledge graph traversal with traditional vector similarity search. While vector search finds documents with similar semantic meaning, knowledge graph search finds explicit relationships between entities (people, organizations, concepts) that span across multiple documents.

> **TL;DR:** Query → Vector Search (Similarity) + Graph Search (Relationships) → Merge Context → Generate

---

## How It Works

```
┌─────────┐
│  Query  │
└────┬────┘
     │
     ├──▶ Vector Database (Chroma) ──▶ Dense Context
     │                                     │
     │                                     ▼
     │                              ┌──────────────┐    ┌──────────┐    ┌────────┐
     │                              │Merge Contexts│───▶│ Generate │───▶│ Answer │
     │                              └──────────────┘    └──────────┘    └────────┘
     │                                     ▲
     │    Cypher Query                     │
     └──▶ (LLM generated) ──▶ Neo4j DB ────┘
                              (Knowledge Graph)
```

### Step-by-Step Flow

1. **Input Validation** — Query passes through guardrails
2. **Parallel Retrieval**:
   - **Vector Search**: Performs standard similarity search in ChromaDB to find semantically relevant text chunks.
   - **Graph Search**: 
     - The LLM translates the natural language query into a Cypher query (using the graph schema for context).
     - The Cypher query is executed against the Neo4j database to find related nodes and edges.
3. **Context Merging** — The raw text chunks and the structured graph relationships are concatenated into a single hybrid context block.
4. **Generation** — The LLM generates a response, explicitly instructed to use the graph connections to "connect the dots" across the text chunks.
5. **Output Validation** — Response passes through output guardrails

### When to Use

| ✅ Good For | ❌ Not Ideal For |
|---|---|
| Deeply connected enterprise data (organizational structures, finance) | Simple knowledge bases without strong entity relationships |
| Multi-hop questions ("Who manages the person who approved X?") | Systems where maintaining a separate Neo4j instance is not viable |
| Compliance and legal documents with cross-references | Quick prototypes |
| When vector search fails to capture explicit structural relationships | Queries focused entirely on sentiment or abstract concepts |

---

## Requirements and Installation

Graph RAG requires an active Neo4j database and additional Python dependencies.

1. **Install dependencies:**
```bash
uv pip install 'modular-rag[graph]'
# This installs langchain-neo4j and neo4j driver
```

2. **Start a Neo4j instance (Docker is easiest):**
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

---

## Configuration

File: `config/architectures/graph_rag.yaml`

```yaml
# Number of documents from vector search
top_k: 5

# Neo4j graph database settings
graph:
  enabled: true
  uri: "bolt://localhost:7687"
  username: "neo4j"
  # password: set via NEO4J_PASSWORD env var
```

### Environment Variables

You must export your Neo4j password before running:
```bash
export NEO4J_PASSWORD="your_password"
```

---

## Testing

```bash
# 1. Start with Graph RAG in verbose mode
uv run main.py --arch graph_rag --verbose

# 2. Ask a relationship-based question
You: Who are the key competitors to Acme Corp mentioned in the documents?
```

**What to verify:**
- Verbose output shows successful connection to Neo4j
- Verbose output displays the generated Cypher query
- The final context block sent to the LLM contains both "Document Context" and "Knowledge Graph Context"
- The generated answer explicitly references relationships that are not obvious from single text chunks alone

---

## Entity Extraction (Ingestion Phase)

Graph RAG is only as good as the data in the graph. During the ingestion phase (`uv run main.py --ingest`), the Graph RAG architecture intercepts the documents and runs an entity extraction pass:

1. The LLM reads chunks of text.
2. It is prompted to extract entities (Nodes) and relationships (Edges).
3. It outputs Cypher `CREATE` statements (e.g., `CREATE (a:Person {name: 'John'})-[:WORKS_AT]->(b:Company {name: 'Acme'})`).
4. These statements are executed against Neo4j to build the graph.

*Note: In production, you would typically use a dedicated information extraction pipeline (like GLiNER or NuNER) rather than relying purely on LLM prompting, but this implementation demonstrates the core pattern.*

---

## Research Papers

| Paper | Year | Venue | Relevance |
|---|---|---|---|
| [GraphRAG: Unlocking LLM discovery on narrative private data](https://arxiv.org/abs/2404.16130) | 2024 | Microsoft | Popularized the "Graph RAG" concept, demonstrating how knowledge graphs significantly improve performance on global, sensemaking questions over private datasets. |

## Implementation

Source: [`src/core/architectures/graph_rag.py`](../../src/core/architectures/graph_rag.py)

Key classes and methods:
- `GraphRAG` — Main architecture class
- `_init_graph_store()` — Connects to Neo4j
- `_query_knowledge_graph()` — Translates NL to Cypher and executes
- `extract_and_store_entities()` — Called during ingestion to build the graph
