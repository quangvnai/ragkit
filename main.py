"""
Entry point for RAG from Scratch — demonstrates all strategies available so far.

Usage:
    python main.py                          # naive RAG (default)
    python main.py --strategy multi_query
    python main.py --strategy rag_fusion
    python main.py --strategy hyde
    python main.py --strategy step_back
    python main.py --strategy decompose_recursive
    python main.py --strategy decompose_parallel
    python main.py --question "Your question here" --strategy rag_fusion
"""

import argparse

from rag.generation.chain import build_rag_chain
from rag.indexing.loader import load_web
from rag.indexing.splitter import tiktoken_splitter
from rag.indexing.vectorstore import build_chroma
from rag.retrieval import (
    decomposition_chain,
    hyde_retriever,
    multi_query_retriever,
    rag_fusion_retriever,
    step_back_chain,
)

SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CSS_CLASSES = ["post-content", "post-title", "post-header"]
DEFAULT_QUESTION = "What is task decomposition for LLM agents?"

STRATEGIES = {
    "naive",
    "multi_query",
    "rag_fusion",
    "hyde",
    "step_back",
    "decompose_recursive",
    "decompose_parallel",
}


def build_vectorstore():
    docs = load_web(SOURCE_URL, css_classes=CSS_CLASSES)
    splits = tiktoken_splitter().split_documents(docs)
    return build_chroma(splits)


def run(question: str, strategy: str) -> str:
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever()

    if strategy == "naive":
        chain = build_rag_chain(retriever)
        return chain.invoke(question)

    if strategy == "multi_query":
        chain = build_rag_chain(multi_query_retriever(retriever))
        return chain.invoke(question)

    if strategy == "rag_fusion":
        chain = build_rag_chain(rag_fusion_retriever(retriever))
        return chain.invoke(question)

    if strategy == "hyde":
        chain = build_rag_chain(hyde_retriever(retriever))
        return chain.invoke(question)

    if strategy == "step_back":
        return step_back_chain(retriever).invoke(question)

    if strategy == "decompose_recursive":
        return decomposition_chain(retriever, mode="recursive").invoke(question)

    if strategy == "decompose_parallel":
        return decomposition_chain(retriever, mode="parallel").invoke(question)

    raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {sorted(STRATEGIES)}")


def main():
    parser = argparse.ArgumentParser(description="RAG from Scratch — strategy runner")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--strategy", default="naive", choices=sorted(STRATEGIES))
    args = parser.parse_args()

    print(f"Strategy : {args.strategy}")
    print(f"Question : {args.question}")
    print("-" * 60)
    answer = run(args.question, args.strategy)
    print(answer)


if __name__ == "__main__":
    main()
