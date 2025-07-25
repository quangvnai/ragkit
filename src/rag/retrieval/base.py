from langchain_core.load import dumps, loads
from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_unique_union(documents: list[list[Document]]) -> list[Document]:
    """Deduplicate across multiple retrieval result lists."""
    flattened = [dumps(doc) for sublist in documents for doc in sublist]
    return [loads(doc) for doc in set(flattened)]
