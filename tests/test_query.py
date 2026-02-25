"""Tests for query and retrieval logic."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.query import build_prompt, search_and_answer, search_with_scores


def test_build_prompt_includes_query():
    """The prompt should contain the user's query."""
    docs = [Document(page_content="test content", metadata={"source": "test.md"})]
    prompt = build_prompt("my question", docs)
    assert "my question" in prompt


def test_build_prompt_includes_doc_content():
    """The prompt should contain the retrieved document content."""
    docs = [Document(page_content="blockchain farming", metadata={"source": "test.md"})]
    prompt = build_prompt("question", docs)
    assert "blockchain farming" in prompt


def test_build_prompt_includes_source():
    """The prompt should include the source file path."""
    docs = [Document(page_content="content", metadata={"source": "/path/to/doc.md"})]
    prompt = build_prompt("question", docs)
    assert "/path/to/doc.md" in prompt


@patch("src.query.ChatOpenAI")
def test_search_and_answer_returns_result(mock_llm_cls):
    """search_and_answer should return answer text and source documents."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="the answer")
    mock_llm_cls.return_value = mock_llm

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        Document(page_content="relevant info", metadata={"source": "doc.md"})
    ]

    answer, sources = search_and_answer(mock_store, "test query")
    assert answer == "the answer"
    assert len(sources) == 1
    assert sources[0].metadata["source"] == "doc.md"


@patch("src.query.ChatOpenAI")
def test_search_with_scores_returns_scores(mock_llm_cls):
    """search_with_scores should return answer text and (doc, score) pairs."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="the answer")
    mock_llm_cls.return_value = mock_llm

    doc = Document(page_content="relevant info", metadata={"source": "doc.md"})
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [(doc, 0.85)]

    answer, results = search_with_scores(mock_store, "test query")
    assert answer == "the answer"
    assert len(results) == 1
    assert results[0][0].metadata["source"] == "doc.md"
    assert results[0][1] == 0.85
