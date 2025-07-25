from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.config import settings
from rag.generation.prompts import RAG_MULTI_QUERY
from rag.retrieval.base import get_unique_union


def multi_query_retriever(
    retriever: BaseRetriever,
    llm=None,
) -> Runnable:
    """Return a Runnable[str, list[Document]] that fans out to 5 query variants.

    The LLM rephrases the original question 5 ways, each variant retrieves
    its own doc set, and the results are deduplicated via get_unique_union.
    Drop-in replacement for any base retriever inside build_rag_chain.
    """
    llm = llm or settings.get_llm()
    generate_queries = (
        {"question": RunnablePassthrough()}
        | RAG_MULTI_QUERY
        | llm
        | StrOutputParser()
        | (lambda x: [q for q in x.split("\n") if q.strip()])
    )
    return generate_queries | retriever.map() | get_unique_union
