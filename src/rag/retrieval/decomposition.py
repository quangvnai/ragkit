from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from rag.config import settings
from rag.generation.prompts import (
    RAG_DECOMPOSE,
    RAG_DECOMPOSE_RECURSIVE,
    RAG_DEFAULT,
    RAG_SYNTHESIS,
)
from rag.retrieval.base import format_docs


def _format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def decomposition_chain(
    retriever: BaseRetriever,
    llm=None,
    mode: Literal["recursive", "parallel"] = "recursive",
) -> Runnable:
    """Return a Runnable[str, str] that answers via question decomposition.

    recursive: each sub-answer is fed as context to the next sub-question
               (builds up Q&A history — Part 7a).
    parallel:  all sub-questions answered independently, then synthesised
               into a final answer (Part 7b).
    """
    llm = llm or settings.get_llm()
    generate_sub_questions = (
        {"question": RunnableLambda(lambda x: x)}
        | RAG_DECOMPOSE
        | llm
        | StrOutputParser()
        | (lambda x: [q for q in x.split("\n") if q.strip()])
    )

    if mode == "recursive":
        def _run(question: str) -> str:
            sub_questions = generate_sub_questions.invoke(question)
            q_a_pairs = ""
            for sub_q in sub_questions:
                context = format_docs(retriever.invoke(sub_q))
                answer = (RAG_DECOMPOSE_RECURSIVE | llm | StrOutputParser()).invoke(
                    {"question": sub_q, "q_a_pairs": q_a_pairs, "context": context}
                )
                q_a_pairs += "\n---\n" + _format_qa_pair(sub_q, answer)
            return q_a_pairs

        return RunnableLambda(_run)

    # parallel mode
    def _run_parallel(question: str) -> str:
        sub_questions = generate_sub_questions.invoke(question)
        qa_parts = []
        for sub_q in sub_questions:
            context = format_docs(retriever.invoke(sub_q))
            answer = (RAG_DEFAULT | llm | StrOutputParser()).invoke(
                {"question": sub_q, "context": context}
            )
            qa_parts.append(f"Q: {sub_q}\nA: {answer}")
        return (RAG_SYNTHESIS | llm | StrOutputParser()).invoke(
            {"question": question, "context": "\n\n".join(qa_parts)}
        )

    return RunnableLambda(_run_parallel)
