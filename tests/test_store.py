"""Tests for vector store operations."""
from unittest.mock import patch, MagicMock
from src.store import create_store, load_store


@patch("src.store.Chroma")
@patch("src.store.OpenAIEmbeddings")
def test_create_store_calls_chroma(mock_embeddings_cls, mock_chroma_cls):
    """create_store should call Chroma.from_documents with the right args."""
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings
    mock_chroma_cls.from_documents.return_value = MagicMock()

    docs = [MagicMock()]
    create_store(docs, persist_dir="/tmp/test_chroma")

    mock_chroma_cls.from_documents.assert_called_once_with(
        documents=docs,
        embedding=mock_embeddings,
        persist_directory="/tmp/test_chroma",
    )


@patch("src.store.Chroma")
@patch("src.store.OpenAIEmbeddings")
def test_load_store_calls_chroma(mock_embeddings_cls, mock_chroma_cls):
    """load_store should initialize Chroma with persist_directory."""
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings

    load_store(persist_dir="/tmp/test_chroma")

    mock_chroma_cls.assert_called_once_with(
        persist_directory="/tmp/test_chroma",
        embedding_function=mock_embeddings,
    )
