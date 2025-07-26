from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from rag.config import settings
from rag.generation.prompts import RAG_STEP_BACK, RAG_STEP_BACK_RESPONSE
from rag.retrieval.base import format_docs


def step_back_chain(
    retriever: BaseRetriever,
    llm=None,
) -> Runnable:
    """Return a Runnable[str, str] implementing the Step-Back prompting strategy.

    The LLM generates a more abstract "step-back" question; docs are retrieved
    for BOTH the original question and the step-back question, then both
    contexts are fed together into generation.
    """
    llm = llm or settings.get_llm()
    generate_step_back = RAG_STEP_BACK | llm | StrOutputParser()

    def _run(question: str) -> str:
        step_back_question = generate_step_back.invoke({"question": question})
        normal_context = format_docs(retriever.invoke(question))
        step_back_context = format_docs(retriever.invoke(step_back_question))
        return (RAG_STEP_BACK_RESPONSE | llm | StrOutputParser()).invoke(
            {
                "question": question,
                "normal_context": normal_context,
                "step_back_context": step_back_context,
            }
        )

    return RunnableLambda(_run)
