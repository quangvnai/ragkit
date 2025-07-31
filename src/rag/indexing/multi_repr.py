import uuid

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import settings
from rag.indexing.vectorstore import build_chroma

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the following document concisely:\n\n{doc}"
)

_ID_KEY = "doc_id"


def build_multi_vector_retriever(
    docs: list[Document],
    llm=None,
    persist_dir: str | None = None,
) -> MultiVectorRetriever:
    """Index docs by their LLM-generated summaries; return full docs at query time.

    Summaries are stored in ChromaDB (retrieved by similarity).
    Full docs are kept in an InMemoryByteStore keyed by UUID.
    MultiVectorRetriever ties the two stores: similarity search over summaries,
    but the caller receives the original full documents.

    persist_dir=None  → in-memory Chroma (notebook use)
    persist_dir=<path> → disk-backed Chroma (summaries survive restarts)
    """
    llm = llm or settings.get_llm()

    summarize_chain = (
        {"doc": lambda x: x.page_content}
        | _SUMMARIZE_PROMPT
        | llm
        | StrOutputParser()
    )
    summaries: list[str] = summarize_chain.batch(docs, {"max_concurrency": 5})

    vectorstore = build_chroma(
        docs=[],
        collection="summaries",
        persist_dir=persist_dir,
    )
    store = InMemoryByteStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=_ID_KEY,
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]
    summary_docs = [
        Document(page_content=s, metadata={_ID_KEY: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever
