from langchain_core.retrievers import BaseRetriever


def cohere_rerank_retriever(
    base_retriever: BaseRetriever,
    top_n: int = 3,
    fetch_k: int = 10,
) -> BaseRetriever:
    """Wrap base_retriever with Cohere Re-Rank to improve result quality.

    Fetches fetch_k candidates from base_retriever first, then calls the
    Cohere Rerank API to score and return only the top_n most relevant docs.

    Requires the cohere package and a COHERE_API_KEY env var (or set in .env).
    Install: pip install 'rag-from-scratch[rerank]'  or  pip install cohere

    If base_retriever is a VectorStoreRetriever, its search_kwargs['k'] is
    updated to fetch_k automatically so enough candidates are fetched.
    """
    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CohereRerank
    except ImportError as exc:
        raise ImportError(
            "langchain_classic is required for Cohere re-ranking.\n"
            "It should be installed as part of langchain. Run:\n"
            "    pip install langchain"
        ) from exc

    # Bump k so enough candidates reach the reranker
    if hasattr(base_retriever, "search_kwargs"):
        base_retriever.search_kwargs = {
            **base_retriever.search_kwargs,
            "k": fetch_k,
        }

    try:
        compressor = CohereRerank(top_n=top_n)
    except ImportError as exc:
        raise ImportError(
            "The cohere package is required for Cohere re-ranking.\n"
            "Install it with:  pip install 'rag-from-scratch[rerank]'\n"
            "or directly:      pip install cohere"
        ) from exc

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
