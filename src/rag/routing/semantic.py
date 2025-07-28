import numpy as np
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag.config import settings


def build_semantic_router(
    prompt_map: dict[str, str],
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
) -> Runnable:
    """
    Routes a query to the most semantically similar prompt template.

    prompt_map = {"physics": physics_template_str, "math": math_template_str}
    Templates must contain a `{query}` placeholder.
    Returns a Runnable[str, str] that produces a generated answer.
    """
    embeddings = embeddings or OpenAIEmbeddings(model=settings.default_embeddings)
    llm = llm or ChatOpenAI(model=settings.default_llm, temperature=0)

    templates = list(prompt_map.values())
    # Pre-embed all prompt templates at construction time
    prompt_embeddings = embeddings.embed_documents(templates)

    def _route(input: dict) -> PromptTemplate:
        query_embedding = embeddings.embed_query(input["query"])
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        best_idx = int(np.argmax(similarity))
        return PromptTemplate.from_template(templates[best_idx])

    return (
        {"query": RunnablePassthrough()}
        | RunnableLambda(_route)
        | llm
        | StrOutputParser()
    )
