from typing import Type

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag.config import settings

_DEFAULT_SYSTEM = (
    "You are an expert at converting user questions into database queries. "
    "Given a question, return a database query optimized to retrieve the most relevant results. "
    "If there are acronyms or words you are not familiar with, do not try to rephrase them."
)


def build_query_analyzer(
    schema: Type[BaseModel],
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
) -> Runnable:
    """
    Returns a chain: str → schema instance.

    The schema (e.g. TutorialSearch) is defined by the caller and kept out of
    this module so the analyzer stays schema-agnostic. Feed the output into a
    vectorstore self-query retriever to apply metadata filters.
    """
    llm = llm or ChatOpenAI(model=settings.default_llm, temperature=0)
    system = system_prompt or _DEFAULT_SYSTEM

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    return prompt | llm.with_structured_output(schema)
