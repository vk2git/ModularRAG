# Advanced RAG

**Production-grade retrieval quality.** Enhances the basic RAG pipeline with three key improvements: LLM-powered query rewriting, hybrid search (vector + keyword), and cross-encoder reranking.

> **TL;DR:** Query → Rewrite → Hybrid Retrieve → Rerank → Generate with Context → Answer

---

## How It Works

```
┌─────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
│  Query  │───▶│ Rewrite  │───▶│ Hybrid Search │───▶│ Rerank   │───▶│ Generate │───▶│ Answer │
│         │    │ (LLM)    │    │ Vector + BM25 │    │ (Cross-  │    │ (LLM)    │    │        │
└─────────┘    └──────────┘    └───────────────┘    │ Encoder) │    └──────────┘    └────────┘
                                                    └──────────┘
```

### Step-by-Step Flow

1. **Input Validation** — Query passes through guardrails (injection detection, PII redaction)
2. **Query Rewriting** — The LLM reformulates the raw query into a search-optimized version, using conversation history for context. For example, "Tell me about that policy" becomes "What is the company return and refund policy?" This dramatically improves retrieval recall.
3. **Hybrid Retrieval** — Runs two searches in parallel:
   - **Vector search**: Semantic similarity using embeddings (captures meaning)
   - **BM25 keyword search**: Exact term matching (captures specific terms the embedding might miss)
   - Results are merged with configurable weights (default: 50/50)
4. **Reranking** — A cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) re-scores every retrieved document against the original query. This is more accurate than vector similarity alone because it processes query and document together.
5. **Generation** — LLM generates a response grounded in the top reranked documents. If no documents are found, falls back to general chat mode.
6. **Output Validation** — Response passes through output guardrails
7. **Memory** — Interaction is saved to chat history

### When to Use

| ✅ Good For | ❌ Not Ideal For |
|---|---|
| Production systems needing high retrieval quality | Simple POCs where speed matters more than precision |
| Large document collections (1000+ pages) | Tiny document sets where basic search suffices |
| Queries using different terminology than docs | When you need reasoning across multiple retrieval passes |
| Multi-turn conversations with context | Systems with strict latency budgets (reranking adds ~200ms) |
| Documents with both technical and natural language | When reranker models can't be installed (limited environments) |

---

## Configuration

File: `config/architectures/advanced.yaml`

```yaml
# Enable/disable query rewriting before retrieval
query_rewrite: true

# Retriever strategy: basic, hybrid, or reranked
retriever_strategy: "reranked"

# Number of final documents after reranking
top_k: 5
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `query_rewrite` | `true` | Enable LLM-based query reformulation. Disable for lower latency. |
| `retriever_strategy` | `"reranked"` | `basic` = vector only, `hybrid` = vector + BM25, `reranked` = hybrid + cross-encoder |
| `top_k` | `5` | Number of documents to retrieve (before or after reranking, depending on strategy) |

### Global Settings That Affect Advanced RAG

| Setting (in `settings.yaml`) | Description |
|---|---|
| `reranker.provider` | `cross_encoder` (free, local) or `cohere` (API key required) |
| `reranker.cross_encoder_model` | Cross-encoder model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `reranker.top_n` | Number of documents to keep after reranking |
| `retriever.hybrid_weights` | `[vector_weight, bm25_weight]` for hybrid mode (default: `[0.5, 0.5]`) |

---

## Testing

```bash
# 1. Start with Advanced RAG
uv run main.py --arch advanced

# 2. Test query rewriting (verbose shows the rewritten query)
uv run main.py --arch advanced --verbose

# 3. Ask a question using informal/different language than your docs
You: what happens when I return stuff?
# Verbose output should show the rewritten query is more specific

# 4. Test with a multi-turn conversation
You: What is the return policy?
You: How long do I have?
# The rewriter should use chat history to understand "how long" refers to returns

# 5. Test fallback (when reranker isn't installed)
# Advanced RAG gracefully falls back to basic retriever if reranking fails
```

**What to verify:**
- Query rewriting improves retrieval relevance (check verbose output)
- Hybrid search finds documents that pure vector search misses
- Reranking changes document order (most relevant first)
- Graceful fallback to basic retriever if reranking dependencies are missing
- Chat history is maintained across turns

---

## Retriever Strategies Explained

### Basic (`"basic"`)
Standard vector similarity search. Same as Naive RAG's retrieval.

### Hybrid (`"hybrid"`)
Combines vector search with BM25 keyword search using Ensemble Retriever:
```
Score = (vector_weight × vector_score) + (bm25_weight × bm25_score)
```
Best when documents use domain-specific terms that embeddings might not capture well.

### Reranked (`"reranked"`)
Adds a cross-encoder on top of hybrid search. The cross-encoder processes each (query, document) pair together, producing much more accurate relevance scores than embedding similarity alone.

**Performance note:** Reranking adds ~100-500ms depending on the number of documents and model size. For latency-sensitive applications, consider using `"hybrid"` without reranking.

---

## Limitations

- **Higher latency** — Query rewriting, hybrid search, and reranking each add processing time
- **Reranker dependency** — Cross-encoder reranking requires `sentence-transformers` (install with `uv pip install 'modular-rag[rerank]'`)
- **Single retrieval pass** — Still only retrieves once (no retry loop like Corrective RAG)
- **No relevance verification** — Doesn't check if retrieved docs are actually relevant (unlike Corrective RAG)

For self-correcting retrieval, see [Corrective RAG](./corrective-rag.md). For hallucination-aware generation, see [Self-RAG](./self-rag.md).

---

## Research Papers

| Paper | Year | Venue | Relevance |
|---|---|---|---|
| [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283) | 2023 | EMNLP | Demonstrates how LLM-based query rewriting improves RAG retrieval quality. Core technique used in this architecture. |
| [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) | 2020 | SIGIR | Foundational work on late-interaction reranking models. The cross-encoder approach used here is a simpler variant of this idea. |
| [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | 2009 | SIGIR | The hybrid search merging strategy used by LangChain's EnsembleRetriever. |
| [A Comprehensive Survey of RAG](https://arxiv.org/abs/2410.12837) | 2024 | arXiv | Positions Advanced RAG as the production-ready evolution of Naive RAG, with query transformation and reranking as key improvements. |

---

## Implementation

Source: [`src/core/architectures/advanced.py`](../../src/core/architectures/advanced.py)

Key classes and methods:
- `AdvancedRAG` — Main architecture class
- `run()` — Entry point for query processing
- `_rewrite_query()` — LLM-based query reformulation
- `_generate_with_context()` — Generation with retrieved context
- `_generate_general()` — Fallback generation without context

Related components:
- [`src/core/components/retriever.py`](../../src/core/components/retriever.py) — `RetrieverFactory` (basic/hybrid/reranked)
- [`src/core/components/reranker.py`](../../src/core/components/reranker.py) — `RerankerFactory` (cross-encoder/cohere)
