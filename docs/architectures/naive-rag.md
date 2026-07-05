# Naive RAG

**The foundational RAG pattern.** A simple linear pipeline that retrieves relevant documents via vector similarity search and uses them as context for LLM generation.

> **TL;DR:** Query → Embed → Retrieve Top-K → Generate with Context → Answer

---

## How It Works

```
┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────┐    ┌────────┐
│  Query  │───▶│  Embed    │───▶│ Retrieve     │───▶│ Generate │───▶│ Answer │
│         │    │  (vector) │    │ Top-K docs   │    │ (LLM)    │    │        │
└─────────┘    └───────────┘    └──────────────┘    └──────────┘    └────────┘
```

### Step-by-Step Flow

1. **Input Validation** - Query passes through guardrails (injection detection, PII redaction)
2. **Embedding** - Query is converted to a vector using the configured embedding model
3. **Retrieval** - Vector store returns top-K documents ranked by similarity score
4. **Score Filtering** - Documents below a relevance threshold are discarded
5. **Generation** - If relevant docs exist, LLM generates an answer grounded in context. If no docs pass the threshold, falls back to general chat (LLM's own knowledge)
6. **Output Validation** - Response passes through output guardrails
7. **Memory** - Interaction is saved to chat history

### When to Use

| ✅ Good For | ❌ Not Ideal For |
|---|---|
| Quick prototypes and POCs | Complex multi-hop questions |
| Simple factual Q&A | Questions requiring reasoning across documents |
| Small document collections (< 1000 pages) | High-stakes accuracy requirements |
| Low-latency applications | Queries where retrieval quality varies |
| Learning RAG fundamentals | Enterprise production systems |

---

## Configuration

File: `config/architectures/naive.yaml`

```yaml
# Score threshold for document relevance
# Lower = stricter filtering (cosine distance: 0 = identical, 2 = opposite)
score_threshold: 1.5

# Number of documents to retrieve
top_k: 3

# Fall back to LLM's own knowledge when no relevant docs found
fallback_to_general_chat: true
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `score_threshold` | `1.5` | Maximum distance score to consider a doc relevant. Lower = stricter. |
| `top_k` | `3` | Number of documents to retrieve from vector store |
| `fallback_to_general_chat` | `true` | If no docs pass threshold, use LLM's training data |

---

## Testing

```bash
# 1. Start with Naive RAG (default)
uv run main.py --arch naive

# 2. Test with a simple question about your documents
You: What is the return policy?

# 3. Test fallback (ask something NOT in your documents)
You: What is the capital of France?

# 4. Test with verbose mode to see scores
uv run main.py --arch naive --verbose
```

**What to verify:**
- Documents with scores below threshold are used as context
- Documents above threshold are filtered out
- Fallback to general chat works when no docs match
- Chat history is maintained across turns

---

## Limitations

- **No query optimization** - Uses the raw user query for retrieval, which may not match document terminology
- **No relevance verification** - Trusts the vector similarity score without LLM-based grading
- **Single retrieval pass** - No retry or fallback if the first retrieval misses
- **Keyword blindness** - Pure semantic search may miss exact keyword matches

These limitations are addressed by the more advanced architectures ([Advanced RAG](./advanced-rag.md), [Corrective RAG](./corrective-rag.md)).


---

## Implementation

Source: [`src/core/architectures/naive.py`](../src/core/architectures/naive.py)

Key classes and methods:
- `NaiveRAG` - Main architecture class
- `run()` - Entry point for query processing
- `_run_rag_mode()` - Generation with retrieved context
- `_run_general_mode()` - Fallback generation without context
