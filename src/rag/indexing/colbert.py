from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def build_colbert_retriever(
    docs: list[Document] | None = None,
    index_name: str = "rag-index",
    k: int = 3,
) -> BaseRetriever:
    """Return a ColBERT retriever backed by RAGatouille.

    If docs is provided the collection is indexed under index_name.
    If index_name already exists on disk (and docs is None) the index is loaded.

    Raises ImportError with install instructions if ragatouille is not installed.
    """
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError as exc:
        raise ImportError(
            "ragatouille is required for ColBERT retrieval.\n"
            "Install it with:  pip install 'rag-from-scratch[colbert]'\n"
            "or directly:      pip install ragatouille"
        ) from exc

    if docs is not None:
        rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        rag.index(
            collection=[doc.page_content for doc in docs],
            index_name=index_name,
            max_document_length=180,
            split_documents=True,
        )
    else:
        rag = RAGPretrainedModel.from_index(index_name)

    return rag.as_langchain_retriever(k=k)
