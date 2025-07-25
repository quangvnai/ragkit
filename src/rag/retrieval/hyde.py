from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.config import settings
from rag.generation.prompts import RAG_HYDE


def hyde_retriever(
    retriever: BaseRetriever,
    llm=None,
) -> Runnable:
    """Return a Runnable[str, list[Document]] using hypothetical document embeddings.

    The LLM writes a hypothetical answer passage for the question; that passage
    is embedded and used as the similarity search query instead of the raw question.
    Drop-in replacement for any base retriever inside build_rag_chain.
    """
    llm = llm or settings.get_llm()
    generate_hypothetical_doc = (
        {"question": RunnablePassthrough()}
        | RAG_HYDE
        | llm
        | StrOutputParser()
    )
    return generate_hypothetical_doc | retriever
