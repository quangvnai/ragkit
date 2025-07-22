from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.generation import prompts


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: BaseRetriever,
    prompt=None,
    llm: ChatOpenAI | None = None,
) -> Runnable:
    """Build a naive RAG chain: question → retrieve → prompt → LLM → str.

    Args:
        retriever: Any LangChain retriever (Chroma, MultiVector, etc.).
        prompt: A ChatPromptTemplate with {context} and {question} variables.
                Defaults to prompts.RAG_DEFAULT.
        llm: A ChatOpenAI (or compatible) model. Defaults to settings.get_llm().

    Returns:
        A Runnable that accepts a question string and returns an answer string.
    """
    prompt = prompt or prompts.RAG_DEFAULT
    llm = llm or settings.get_llm()
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
