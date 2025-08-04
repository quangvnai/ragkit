"""Tests for routing strategies — uses mocks to avoid API calls."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.routing.logical import build_logical_router
from rag.routing.semantic import build_semantic_router


# ---------------------------------------------------------------------------
# Logical Router Tests
# ---------------------------------------------------------------------------

def test_logical_router_dispatches_correctly(mocker):
    # Mock LLM and its structured output
    mock_llm = MagicMock()
    
    # Mock the return value of with_structured_output
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # Mock the route result
    # We can use a simple namespace or a mock with the attribute set
    mock_route = MagicMock()
    mock_route.datasource = "python"
    
    # LangChain Runnable calls .invoke()
    mock_structured_llm.invoke.return_value = mock_route
    # Also handle direct call if LangChain uses it
    mock_structured_llm.return_value = mock_route

    # Mock retrievers
    retriever_python = MagicMock()
    retriever_js = MagicMock()
    datasources = {"python": retriever_python, "js": retriever_js}

    router = build_logical_router(datasources, llm=mock_llm)
    
    # Invoke the router
    # Note: build_logical_router returns router | RunnableLambda(_dispatch)
    # The result should be the retriever itself
    result = router.invoke("How do I use decorators in Python?")
    assert result == retriever_python


# ---------------------------------------------------------------------------
# Semantic Router Tests
# ---------------------------------------------------------------------------

def test_semantic_router_selects_best_template(mocker):
    # Mock Embeddings
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
    mock_embeddings.embed_query.return_value = [0.9, 0.1] # Closer to [1.0, 0.0]

    # Mock LLM to return an AIMessage (StrOutputParser expects this)
    mock_llm = MagicMock()
    msg = AIMessage(content="Answer from physics template")
    mock_llm.invoke.return_value = msg
    mock_llm.return_value = msg

    prompt_map = {
        "physics": "Physics context: {query}",
        "math": "Math context: {query}"
    }

    router = build_semantic_router(prompt_map, embeddings=mock_embeddings, llm=mock_llm)
    
    answer = router.invoke("What is gravity?")
    
    assert answer == "Answer from physics template"
    # Verify embeddings were called
    assert mock_embeddings.embed_query.called
    # Verify LLM was called (either directly or via .invoke)
    assert mock_llm.called
