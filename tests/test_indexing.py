"""Tests for indexing helpers — no API calls required."""

import pytest
from langchain_core.documents import Document

from rag.indexing.splitter import tiktoken_splitter
from rag.config import settings


# ---------------------------------------------------------------------------
# tiktoken_splitter
# ---------------------------------------------------------------------------

def test_tiktoken_splitter_returns_splitter():
    splitter = tiktoken_splitter()
    assert splitter is not None
    assert hasattr(splitter, "split_documents")


def test_tiktoken_splitter_uses_settings_defaults():
    splitter = tiktoken_splitter()
    # RecursiveCharacterTextSplitter stores these internally
    assert splitter._chunk_size == settings.chunk_size
    assert splitter._chunk_overlap == settings.chunk_overlap


def test_tiktoken_splitter_respects_explicit_params():
    splitter = tiktoken_splitter(chunk_size=100, chunk_overlap=10)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 10


def test_tiktoken_splitter_splits_long_document():
    splitter = tiktoken_splitter(chunk_size=20, chunk_overlap=0)
    long_doc = Document(page_content=" ".join(["word"] * 200))
    splits = splitter.split_documents([long_doc])
    assert len(splits) > 1
    for chunk in splits:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) > 0


def test_tiktoken_splitter_short_doc_stays_single_chunk():
    splitter = tiktoken_splitter(chunk_size=300, chunk_overlap=50)
    short_doc = Document(page_content="Short document.")
    splits = splitter.split_documents([short_doc])
    assert len(splits) == 1
    assert splits[0].page_content == "Short document."


def test_tiktoken_splitter_propagates_metadata():
    splitter = tiktoken_splitter(chunk_size=20, chunk_overlap=0)
    doc = Document(page_content=" ".join(["word"] * 100), metadata={"source": "test"})
    splits = splitter.split_documents([doc])
    for chunk in splits:
        assert chunk.metadata.get("source") == "test"
