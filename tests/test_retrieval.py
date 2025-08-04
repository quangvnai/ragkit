"""Tests for pure retrieval functions — no API calls required."""

import pytest
from langchain_core.documents import Document

from rag.retrieval.base import get_unique_union
from rag.retrieval.rag_fusion import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def four_docs():
    return [
        Document(page_content="alpha"),
        Document(page_content="beta"),
        Document(page_content="gamma"),
        Document(page_content="delta"),
    ]


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

def test_rrf_top_doc_appears_in_both_lists(four_docs):
    doc1, doc2, doc3, doc4 = four_docs
    # doc1 ranks first in BOTH lists → highest combined RRF score
    result = reciprocal_rank_fusion([[doc1, doc2, doc3], [doc1, doc4, doc2]])
    top_doc, top_score = result[0]
    assert top_doc.page_content == doc1.page_content


def test_rrf_single_appearance_outranked_by_double(four_docs):
    doc1, doc2, doc3, doc4 = four_docs
    # doc2 appears in both lists, doc3 only in one → doc2 beats doc3
    result = reciprocal_rank_fusion([[doc1, doc2, doc3], [doc4, doc2]])
    scored = {d.page_content: s for d, s in result}
    assert scored["beta"] > scored["gamma"]


def test_rrf_scores_are_positive(four_docs):
    doc1, doc2, doc3, doc4 = four_docs
    result = reciprocal_rank_fusion([[doc1, doc2], [doc3, doc4]])
    for _, score in result:
        assert score > 0


def test_rrf_returns_all_unique_docs(four_docs):
    doc1, doc2, doc3, doc4 = four_docs
    result = reciprocal_rank_fusion([[doc1, doc2], [doc3, doc4]])
    assert len(result) == 4


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


# ---------------------------------------------------------------------------
# get_unique_union
# ---------------------------------------------------------------------------

def test_get_unique_union_deduplicates(four_docs):
    doc1, doc2, doc3, _ = four_docs
    # doc2 appears in both sub-lists
    result = get_unique_union([[doc1, doc2], [doc2, doc3]])
    assert len(result) == 3


def test_get_unique_union_preserves_all_when_no_overlap(four_docs):
    doc1, doc2, doc3, doc4 = four_docs
    result = get_unique_union([[doc1, doc2], [doc3, doc4]])
    assert len(result) == 4


def test_get_unique_union_handles_empty_sublists(four_docs):
    doc1, doc2, *_ = four_docs
    result = get_unique_union([[doc1, doc2], []])
    assert len(result) == 2


def test_get_unique_union_returns_documents(four_docs):
    doc1, doc2, *_ = four_docs
    result = get_unique_union([[doc1], [doc2]])
    assert all(isinstance(d, Document) for d in result)
