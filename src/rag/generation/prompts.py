from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ── Part 4: Basic RAG ─────────────────────────────────────────────────────────

RAG_DEFAULT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}"""
)

# ── Part 5: Multi-Query ───────────────────────────────────────────────────────

RAG_MULTI_QUERY = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five \
different versions of the given user question to retrieve relevant documents from a vector \
database. By generating multiple perspectives on the user question, your goal is to help \
the user overcome some of the limitations of the distance-based similarity search. \
Provide these alternative questions separated by newlines. Original question: {question}"""
)

# ── Part 6: RAG-Fusion ────────────────────────────────────────────────────────

RAG_FUSION = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to: {question}

Output (4 queries):"""
)

# ── Part 7: Decomposition ─────────────────────────────────────────────────────

RAG_DECOMPOSE = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple sub-questions related to an input question.

The goal is to break down the input into a set of sub-problems / sub-questions that can be \
answered in isolation.

Generate multiple search queries related to: {question}

Output (3 queries):"""
)

# Used in the recursive decomposition mode: each sub-question receives prior Q&A context.
RAG_DECOMPOSE_RECURSIVE = ChatPromptTemplate.from_template(
    """Here is the question you need to answer:

---
{question}
---

Here is any available background question + answer pairs:

---
{q_a_pairs}
---

Here is additional context relevant to the question:

---
{context}
---

Use the above context and any background question + answer pairs to answer the question: {question}"""
)

# Used in the parallel decomposition mode: synthesises individual Q&A pairs into one answer.
RAG_SYNTHESIS = ChatPromptTemplate.from_template(
    """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}"""
)

# ── Part 8: Step-Back ─────────────────────────────────────────────────────────

_STEP_BACK_EXAMPLES = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
]

_step_back_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)

_step_back_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_step_back_example_prompt,
    examples=_STEP_BACK_EXAMPLES,
)

RAG_STEP_BACK = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at world knowledge. Your task is to step back and paraphrase "
            "a question to a more generic step-back question, which is easier to answer. "
            "Here are a few examples:",
        ),
        _step_back_few_shot,
        ("user", "{question}"),
    ]
)

# Response prompt for step-back: merges normal + step-back context.
RAG_STEP_BACK_RESPONSE = ChatPromptTemplate.from_template(
    """You are an expert of world knowledge. I am going to ask you a question. Your response \
should be comprehensive and not contradicted with the following context if they are relevant. \
Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
)

# ── Part 9: HyDE ─────────────────────────────────────────────────────────────

RAG_HYDE = ChatPromptTemplate.from_template(
    """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
)
