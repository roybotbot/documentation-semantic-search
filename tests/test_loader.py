"""Tests for document loading and chunking."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.loader import chunk_documents, load_documents


def test_chunk_documents_splits_large_text():
    """A document larger than chunk_size should be split into multiple chunks."""
    long_text = "word " * 500  # ~2500 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.md"})]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_chunk_documents_preserves_metadata():
    """Chunked documents should keep their source metadata."""
    docs = [Document(page_content="short text", metadata={"source": "test.md"})]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert chunks[0].metadata["source"] == "test.md"


def test_chunk_documents_small_text_stays_intact():
    """A document smaller than chunk_size should not be split."""
    docs = [Document(page_content="small", metadata={"source": "test.md"})]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1


def test_load_documents_invalid_path():
    """Loading from a nonexistent path should raise ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        load_documents("/nonexistent/path")


def test_load_documents_not_a_directory(tmp_path):
    """Loading from a file path should raise ValueError."""
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="not a directory"):
        load_documents(str(f))
