from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from rag.config import settings


def build_logical_router(
    datasources: dict[str, BaseRetriever],
    llm: BaseChatModel | None = None,
) -> Runnable:
    """
    Classifies an input question using LLM structured output and dispatches
    it to the matching retriever from `datasources`.

    datasources = {"python_docs": retriever_a, "js_docs": retriever_b}
    Returns a Runnable[str, list[Document]].
    """
    llm = llm or ChatOpenAI(model=settings.default_llm, temperature=0)

    keys = list(datasources.keys())
    # Build a Literal type dynamically from datasource keys
    datasource_literal = Literal[tuple(keys)]  # type: ignore[valid-type]

    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""
        datasource: datasource_literal = Field(  # type: ignore[valid-type]
            ...,
            description=(
                "Given a user question, choose which datasource would be most "
                "relevant for answering it."
            ),
        )

    structured_llm = llm.with_structured_output(RouteQuery)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at routing a user question to the appropriate "
                "data source.\n\nAvailable datasources: "
                + ", ".join(keys)
                + "\n\nRoute to the most relevant datasource.",
            ),
            ("human", "{question}"),
        ]
    )

    router = prompt | structured_llm

    def _dispatch(route: RouteQuery):
        return datasources[route.datasource]

    return router | RunnableLambda(_dispatch)
