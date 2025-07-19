from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag.indexing.embedder import get_embeddings


def build_chroma(
    docs: list[Document],
    collection: str = "default",
    persist_dir: str | None = None,
) -> Chroma:
    """Build a Chroma vectorstore from documents.

    persist_dir=None  → in-memory (notebook use, discarded on exit)
    persist_dir=<path> → disk-backed (index once, reuse across runs)
    """
    kwargs: dict = {"collection_name": collection, "embedding_function": get_embeddings()}
    if persist_dir is not None:
        kwargs["persist_directory"] = persist_dir
    return Chroma.from_documents(documents=docs, **kwargs)
