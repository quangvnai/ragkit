from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings


def tiktoken_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a tiktoken-aware recursive splitter using project defaults."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
