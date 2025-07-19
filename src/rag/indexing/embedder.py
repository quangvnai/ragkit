from langchain_openai import OpenAIEmbeddings

from rag.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return an embeddings instance wired to the configured provider."""
    return settings.get_embeddings()
