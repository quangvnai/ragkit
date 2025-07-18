# ragkit

**A modular toolkit covering every RAG technique — from naive to Self-RAG.**

Each technique is an importable, composable module — swap retrieval strategies in one line, run everything locally with Gemma via LM Studio.

---

## Techniques covered

| # | Technique | Category |
|---|-----------|----------|
| 1–4 | Naive RAG (load → split → embed → retrieve → generate) | Fundamentals |
| 5 | Multi-Query | Query Translation |
| 6 | RAG-Fusion + Reciprocal Rank Fusion | Query Translation |
| 7 | Decomposition (recursive & parallel) | Query Translation |
| 8 | Step-Back | Query Translation |
| 9 | HyDE | Query Translation |
| 10 | Logical & Semantic Routing | Routing |
| 11 | Query Structuring for metadata filters | Query Construction |
| 12 | Multi-Representation Indexing | Indexing |
| 13 | RAPTOR | Indexing |
| 14 | ColBERT | Indexing |
| 15 | Re-Ranking (RRF + Cohere) | Retrieval |
| 16–17 | CRAG & Self-RAG | Generation |

## Setup

```bash
git clone https://github.com/quangvnai/ragkit
cd ragkit
pip install -e ".[dev]"
cp .env.example .env   # then edit .env
```

Requires [LM Studio](https://lmstudio.ai) running locally with Gemma loaded, or set `LLM_PROVIDER=openai` in `.env`.
