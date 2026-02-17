"""Vector store operations using ChromaDB."""
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src import config

logger = logging.getLogger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    """Create embeddings model instance."""
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def create_store(docs: list, persist_dir: str = config.CHROMA_DB_DIR) -> Chroma:
    """Create a new vector store from documents.

    Args:
        docs: List of chunked Document objects.
        persist_dir: Directory to persist the vector store.

    Returns:
        Chroma vector store instance.
    """
    logger.info("Creating vector store with %d documents...", len(docs))
    store = Chroma.from_documents(
        documents=docs,
        embedding=_get_embeddings(),
        persist_directory=persist_dir,
    )
    logger.info("Vector store saved to %s", persist_dir)
    return store


def load_store(persist_dir: str = config.CHROMA_DB_DIR) -> Chroma:
    """Load an existing vector store.

    Args:
        persist_dir: Directory where vector store is persisted.

    Returns:
        Chroma vector store instance.
    """
    logger.info("Loading vector store from %s", persist_dir)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings(),
    )
