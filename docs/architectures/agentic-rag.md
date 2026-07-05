# Agentic RAG

**Iterative reasoning.** Treats the LLM as an autonomous agent equipped with tools. Instead of a fixed retrieve-then-generate pipeline, the agent decides *when* to retrieve, *what* to retrieve, and *when* it has enough information to formulate an answer.

> **TL;DR:** LLM Agent ↔ [Tools: Local Search, Web Search] → Answer

---

## How It Works

```
┌─────────┐    ┌─────────────┐
│  Query  │───▶│  LLM Agent  │──▶ Answer
└─────────┘    └──────┬──────┘
                  ▲   │   ▲
                  │   │   │
           Results│   │   │Results
                  │   ▼   │
               ┌───────────┐
               │   Tools   │
               │ - Vector  │
               │ - Web     │
               └───────────┘
```

### Step-by-Step Flow

1. **Input Validation** — Query passes through guardrails
2. **Agent Initialization** — The agent is given a system prompt, the user's query, and a list of callable tools (e.g., `retrieve_documents`, `web_search`).
3. **Reasoning Loop** (powered by LangGraph's ReAct implementation):
   - The agent analyzes the question and decides what information it needs.
   - It calls a tool (e.g., `retrieve_documents("return policy")`).
   - The tool executes and returns context to the agent.
   - The agent reads the context. If it needs more, it might call the tool again with a different query, or try a different tool (e.g., `web_search`).
   - Once it has sufficient information, it generates the final response.
4. **Output Validation** — Response passes through output guardrails

### LangGraph State Machine

This architecture uses the prebuilt `create_react_agent` from LangGraph, which abstracts the complex state machine required for tool calling and reasoning loops.

### When to Use

| ✅ Good For | ❌ Not Ideal For |
|---|---|
| Complex, multi-hop questions ("Who is the CEO of the company that acquired X?") | Simple, direct lookups (overkill) |
| Questions requiring data synthesis from multiple distinct searches | Latency-sensitive applications (can take many LLM calls) |
| Open-ended research queries | Models with poor tool-calling capabilities |
| When the retrieval strategy isn't known upfront | Highly constrained, deterministic environments |

---

## Configuration

File: `config/architectures/agentic.yaml`

```yaml
# Number of documents per retrieval
top_k: 5

# Maximum agent reasoning steps (prevents infinite loops)
max_agent_steps: 10

# Enable web search as an agent tool
web_search: true
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `top_k` | `5` | Documents retrieved per `retrieve_documents` tool call |
| `max_agent_steps` | `10` | Hard limit on the reasoning loop to prevent infinite tool calling loops |
| `web_search` | `true` | Exposes the web search tool to the agent (requires provider config in `settings.yaml`) |

---

## Testing

```bash
# 1. Start with Agentic RAG
uv run main.py --arch agentic

# 2. Test with verbose mode to watch the agent's reasoning process
uv run main.py --arch agentic --verbose

# 3. Ask a multi-hop question that requires two searches
You: What is the relationship between [Entity A in doc 1] and [Entity B in doc 2]?
# Verbose should show the agent performing multiple tool calls

# 4. Ask a question requiring both local and web knowledge
You: Compare our internal policy on X with the current industry standard.
```

**What to verify:**
- The agent successfully calls tools
- The agent correctly decides when to stop searching and answer
- The agent can chain multiple searches together logically

---

## Limitations and Requirements

- **Model Capability** — Agentic RAG *requires* an LLM with strong tool-calling (function calling) capabilities. GPT-4, Claude 3, and Gemini perform well. Small local models (like Mistral 7B) may struggle to format tool calls correctly or get stuck in loops.
- **Latency** — This is the slowest architecture. A complex question might require 4-5 LLM calls before generating the final answer.
- **Cost** — More LLM calls = higher API costs.

---

## Implementation

Source: [`src/core/architectures/agentic.py`](../../src/core/architectures/agentic.py)

Key classes and methods:
- `AgenticRAG` — Main architecture class
- `_build_tools()` — Defines the `retrieve_documents` and `web_search` tools the agent can use
- `_build_graph()` — Instantiates the LangGraph ReAct agent
