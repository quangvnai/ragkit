"""Unified pipeline runner and CLI entry point.

Usage:
    python -m rag.pipeline.runner --question "What is task decomposition?" --strategy rag_fusion
    python -m rag.pipeline.runner --question "What is HyDE?" --strategy hyde --persist-dir ./chroma_db
"""

import argparse

from rag.config import settings
from rag.generation.chain import build_rag_chain
from rag.indexing import build_chroma, load_web, tiktoken_splitter
from rag.retrieval.decomposition import decomposition_chain
from rag.retrieval.hyde import hyde_retriever
from rag.retrieval.multi_query import multi_query_retriever
from rag.retrieval.rag_fusion import rag_fusion_retriever
from rag.retrieval.step_back import step_back_chain

_DEFAULT_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Maps strategy name → (retriever_wrapper | None, needs_rag_chain)
# Stored as callables: _build_*(base_retriever, llm) → Runnable[str, str]
STRATEGIES = {
    "naive",
    "multi_query",
    "rag_fusion",
    "hyde",
    "step_back",
    "decompose",
}


def _build_pipeline(strategy: str, base_retriever, llm):
    """Return a complete Runnable[str, str] for the chosen strategy."""
    if strategy == "naive":
        return build_rag_chain(base_retriever, llm=llm)
    if strategy == "multi_query":
        return build_rag_chain(multi_query_retriever(base_retriever, llm=llm), llm=llm)
    if strategy == "rag_fusion":
        return build_rag_chain(rag_fusion_retriever(base_retriever, llm=llm), llm=llm)
    if strategy == "hyde":
        return build_rag_chain(hyde_retriever(base_retriever, llm=llm), llm=llm)
    if strategy == "step_back":
        return step_back_chain(base_retriever, llm=llm)
    if strategy == "decompose":
        return decomposition_chain(base_retriever, llm=llm)
    raise ValueError(
        f"Unknown strategy {strategy!r}. Valid choices: {sorted(STRATEGIES)}"
    )


def run(
    question: str,
    strategy: str = "naive",
    source_url: str = _DEFAULT_URL,
    persist_dir: str | None = None,
) -> str:
    """Build the chosen RAG pipeline, run it, and return the answer string.

    Args:
        question:    The user question to answer.
        strategy:    One of the STRATEGIES keys (default: "naive").
        source_url:  Web page to load and index. Defaults to Lilian Weng's agent post.
        persist_dir: Path for disk-backed ChromaDB. None = in-memory (re-indexes each run).

    Returns:
        The generated answer string.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Valid choices: {sorted(STRATEGIES)}"
        )

    llm = settings.get_llm()

    docs = load_web(source_url)
    splits = tiktoken_splitter().split_documents(docs)
    vectorstore = build_chroma(splits, persist_dir=persist_dir)
    base_retriever = vectorstore.as_retriever()

    chain = _build_pipeline(strategy, base_retriever, llm)
    return chain.invoke(question)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rag.pipeline.runner",
        description="Run a RAG strategy against a source URL and print the answer.",
    )
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--strategy",
        default="naive",
        choices=sorted(STRATEGIES),
        help="Retrieval strategy (default: naive)",
    )
    parser.add_argument(
        "--source-url",
        default=_DEFAULT_URL,
        help="URL to load and index (default: Lilian Weng agent post)",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory for disk-backed ChromaDB (omit for in-memory)",
    )
    args = parser.parse_args()

    answer = run(
        question=args.question,
        strategy=args.strategy,
        source_url=args.source_url,
        persist_dir=args.persist_dir,
    )
    print(answer)


if __name__ == "__main__":
    _cli()
