from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.config import settings
from rag.generation.prompts import RAG_FUSION


def reciprocal_rank_fusion(
    results: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    """Pure RRF scoring — no LangChain dependency. Score = Σ 1/(rank + k)."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = dumps(doc)
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + k)
            doc_map[key] = doc
    return [
        (doc_map[key], score)
        for key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def rag_fusion_retriever(
    retriever: BaseRetriever,
    llm=None,
    n_queries: int = 4,
) -> Runnable:
    """Return a Runnable[str, list[Document]] using RAG-Fusion.

    The LLM generates n_queries related queries, each retrieves its own docs,
    then results are merged with Reciprocal Rank Fusion (highest-ranked first).
    Drop-in replacement for any base retriever inside build_rag_chain.
    """
    llm = llm or settings.get_llm()
    generate_queries = (
        {"question": RunnablePassthrough()}
        | RAG_FUSION
        | llm
        | StrOutputParser()
        | (lambda x: [q for q in x.split("\n") if q.strip()][:n_queries])
    )
    return (
        generate_queries
        | retriever.map()
        | (lambda results: [doc for doc, _ in reciprocal_rank_fusion(results)])
    )
