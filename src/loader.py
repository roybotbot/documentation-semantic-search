"""Document loading and chunking for markdown files."""
import logging
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.preprocessor import preprocess

logger = logging.getLogger(__name__)


def load_documents(docs_path: str) -> list:
    """Load markdown and mdx files from a directory.

    Args:
        docs_path: Path to directory containing documentation files.

    Returns:
        List of loaded Document objects with preprocessed content.

    Raises:
        ValueError: If path does not exist or is not a directory.
    """
    path = Path(docs_path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {docs_path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {docs_path}")

    all_docs = []
    for ext in config.FILE_EXTENSIONS:
        loader = DirectoryLoader(
            str(path),
            glob=f"**/{ext}",
            loader_cls=UnstructuredMarkdownLoader,
        )
        docs = loader.load()
        logger.info("Loaded %d %s files", len(docs), ext)
        all_docs.extend(docs)

    for doc in all_docs:
        doc.page_content = preprocess(doc.page_content)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


def chunk_documents(
    docs: list,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list:
    """Split documents into chunks for embedding.

    Args:
        docs: List of Document objects to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks", len(docs), len(chunks))
    return chunks
