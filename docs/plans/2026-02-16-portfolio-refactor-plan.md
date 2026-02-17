# Portfolio Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure two scripts into a clean Python package with tests, CI, and a case-study README.

**Architecture:** Extract logic from `load-docs.py` and `query.py` into a `src/` package with separate modules for config, loading, storage, and querying. A thin `cli.py` handles arg parsing. Tests mock all OpenAI calls.

**Tech Stack:** Python 3.12, LangChain, ChromaDB, OpenAI, pytest, ruff, GitHub Actions

---

### Task 1: Project scaffolding and .gitignore

**Files:**
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config.py`

**Step 1: Create .gitignore**

Create `.gitignore`:
```
__pycache__/
*.pyc
.venv/
venv/
chroma_db/
.env
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
```

**Step 2: Create config module**

Create `src/__init__.py` (empty file).

Create `src/config.py`:
```python
"""Central configuration for the semantic search system."""

# Embedding model - OpenAI's smallest and cheapest embedding model.
# 1536 dimensions, good enough for document retrieval.
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM for answer generation
LLM_MODEL = "gpt-5-nano"
LLM_TEMPERATURE = 0

# Chunking parameters
# 1000 chars keeps chunks small enough for precise retrieval
# while preserving enough context to be useful.
CHUNK_SIZE = 1000
# 200 char overlap prevents losing context at chunk boundaries.
CHUNK_OVERLAP = 200

# Number of similar documents to retrieve per query
RETRIEVAL_K = 3

# ChromaDB persistence directory
CHROMA_DB_DIR = "./chroma_db"

# Supported file extensions
FILE_EXTENSIONS = ["*.md", "*.mdx"]
```

**Step 3: Commit**

```bash
git add .gitignore src/__init__.py src/config.py
git commit -m "Add project scaffolding, .gitignore, and config module"
```

---

### Task 2: Extract loader module

**Files:**
- Create: `src/loader.py`
- Create: `tests/test_loader.py`

**Step 1: Write the failing test**

Create `tests/__init__.py` (empty file).

Create `tests/test_loader.py`:
```python
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
```

**Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pip install pytest
pytest tests/test_loader.py -v
```
Expected: FAIL — `src.loader` does not exist yet.

**Step 3: Write the implementation**

Create `src/loader.py`:
```python
"""Document loading and chunking for markdown files."""
import logging
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config

logger = logging.getLogger(__name__)


def load_documents(docs_path: str) -> list:
    """Load markdown and mdx files from a directory.

    Args:
        docs_path: Path to directory containing documentation files.

    Returns:
        List of loaded Document objects.

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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/loader.py tests/__init__.py tests/test_loader.py
git commit -m "Add loader module with chunking logic and tests"
```

---

### Task 3: Extract store module

**Files:**
- Create: `src/store.py`
- Create: `tests/test_store.py`

**Step 1: Write the failing test**

Create `tests/test_store.py`:
```python
"""Tests for vector store operations."""
import pytest
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_store.py -v
```
Expected: FAIL — `src.store` does not exist yet.

**Step 3: Write the implementation**

Create `src/store.py`:
```python
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_store.py -v
```
Expected: All 2 tests PASS.

**Step 5: Install langchain-chroma**

```bash
pip install langchain-chroma
```

**Step 6: Commit**

```bash
git add src/store.py tests/test_store.py
git commit -m "Add vector store module with create/load and tests"
```

---

### Task 4: Extract query module

**Files:**
- Create: `src/query.py`
- Create: `tests/test_query.py`

**Step 1: Write the failing test**

Create `tests/test_query.py`:
```python
"""Tests for query and retrieval logic."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.query import build_prompt, search_and_answer


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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_query.py -v
```
Expected: FAIL — `src.query` does not exist yet.

**Step 3: Write the implementation**

Create `src/query.py`:
```python
"""Query retrieval and answer generation."""
import logging
from langchain_openai import ChatOpenAI
from src import config

logger = logging.getLogger(__name__)


def build_prompt(query: str, docs: list) -> str:
    """Build an LLM prompt from a query and retrieved documents.

    Args:
        query: The user's question.
        docs: Retrieved Document objects for context.

    Returns:
        Formatted prompt string.
    """
    docs_content = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )
    return f"""You are a helpful assistant answering questions about technical documentation.

Use the following context to answer the query. If the answer cannot be found in the context, say so.

Context:
{docs_content}

Query: {query}

Answer:"""


def search_and_answer(store, query: str, k: int = config.RETRIEVAL_K) -> tuple:
    """Search the vector store and generate an answer.

    Args:
        store: Chroma vector store instance.
        query: The user's question.
        k: Number of similar documents to retrieve.

    Returns:
        Tuple of (answer_text, source_documents).
    """
    logger.info("Searching for: %s", query)
    docs = store.similarity_search(query, k=k)
    logger.info("Retrieved %d documents", len(docs))

    prompt = build_prompt(query, docs)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    response = llm.invoke(prompt)

    return response.content, docs
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_query.py -v
```
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/query.py tests/test_query.py
git commit -m "Add query module with prompt building and retrieval, plus tests"
```

---

### Task 5: Create CLI entry point

**Files:**
- Create: `cli.py`
- Delete (after verifying): `load-docs.py`, `query.py` (the old ones)

**Step 1: Write the CLI**

Create `cli.py`:
```python
"""CLI entry point for the semantic search system."""
import argparse
import logging
import os
import sys


def setup_logging(verbose: bool = False):
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )


def check_api_key():
    """Verify OpenAI API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)


def cmd_load(args):
    """Load and index documentation."""
    from src.loader import load_documents, chunk_documents
    from src.store import create_store

    check_api_key()
    docs = load_documents(args.path)
    chunks = chunk_documents(docs)
    create_store(chunks)
    print(f"Indexed {len(docs)} documents ({len(chunks)} chunks).")


def cmd_query(args):
    """Query the indexed documentation."""
    from src.loader import load_documents, chunk_documents
    from src.store import load_store
    from src.query import search_and_answer

    check_api_key()
    store = load_store()
    answer, sources = search_and_answer(store, args.question)

    print(f"\nQuery: {args.question}\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)
    for i, doc in enumerate(sources, 1):
        print(f"\n[{i}] {doc.metadata.get('source', 'Unknown')}")
        print(f"    {doc.page_content[:150]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over documentation using RAG."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load", help="Load and index documentation")
    load_parser.add_argument("path", help="Path to documentation directory")

    query_parser = subparsers.add_parser("query", help="Query indexed documentation")
    query_parser.add_argument("question", help="Question to ask")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "load":
        cmd_load(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it works (dry run, no API call needed)**

```bash
python cli.py --help
python cli.py load --help
python cli.py query --help
```
Expected: Help text prints for each command.

**Step 3: Remove old scripts**

```bash
git rm load-docs.py query.py
```

**Step 4: Commit**

```bash
git add cli.py
git commit -m "Add CLI entry point and remove old scripts"
```

---

### Task 6: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements with pinned versions**

Overwrite `requirements.txt`:
```
langchain==1.2.3
langchain-chroma>=0.2.0
langchain-community==0.4.1
langchain-openai==1.1.7
langchain-text-splitters==1.1.0
chromadb==1.4.0
unstructured==0.18.27
markdown==3.7
pytest>=8.0
ruff>=0.9.0
```

**Step 2: Install new deps**

```bash
source .venv/bin/activate
pip install langchain-chroma pytest ruff
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: All 11 tests PASS.

**Step 4: Run linter**

```bash
ruff check src/ tests/ cli.py
```
Expected: No errors (fix any that come up).

**Step 5: Commit**

```bash
git add requirements.txt
git commit -m "Update requirements with pinned versions, add dev deps"
```

---

### Task 7: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint
        run: ruff check src/ tests/ cli.py

      - name: Test
        run: pytest tests/ -v
```

**Step 2: Remove old Pylint workflow if present**

```bash
ls .github/workflows/ 2>/dev/null
# If pylint.yml exists, remove it:
# git rm .github/workflows/pylint.yml
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI with ruff and pytest"
```

---

### Task 8: Rewrite README

**Files:**
- Modify: `README.md`

**Step 1: Write the new README**

Overwrite `README.md`:
```markdown
# Documentation Semantic Search

Turn any documentation repository into a natural language search system using retrieval-augmented generation (RAG).

## Why

Keyword search fails when users know what they want but not the terminology the docs use. This system converts documentation into vector embeddings and retrieves results by meaning, not string matching.

Built from experience maintaining ~200 pages of developer documentation at Chia Network, where users regularly struggled to find answers that existed in the docs.

## How It Works

```
Markdown files
    │
    ▼
┌──────────┐    ┌──────────────┐    ┌─────────────┐
│  Loader  │───▶│   Chunker    │───▶│  Embeddings │
│ (md/mdx) │    │ (1000 char)  │    │  (OpenAI)   │
└──────────┘    └──────────────┘    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  ChromaDB   │
                                    │ (local)     │
                                    └──────┬──────┘
                                           │
User query ──▶ Embed ──▶ Similarity Search─┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   LLM       │──▶ Answer + Sources
                                    └─────────────┘
```

## Design Decisions

**Chunk size: 1000 characters, 200 overlap.** Small enough for precise retrieval, large enough to preserve context. The overlap prevents losing information at chunk boundaries — a sentence split across two chunks still appears in full in at least one.

**Embedding model: `text-embedding-3-small`.** OpenAI's cheapest embedding model. For documentation search, the difference between this and larger models is marginal — the bottleneck is chunk quality, not embedding precision.

**Local ChromaDB.** No external infrastructure needed. The full Chia docs (~200 files) index in under 5 minutes and the database is a few hundred MB. A hosted vector DB adds complexity without benefit at this scale.

**LangChain for orchestration.** Handles document loading, text splitting, and the retrieval pipeline. Avoids reimplementing standard RAG plumbing.

## Setup

Requires Python 3.8–3.13 and an OpenAI API key.

```bash
git clone https://github.com/yourusername/chia-docs-semantic-search.git
cd chia-docs-semantic-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='your-key-here'
```

## Usage

**Index documentation:**
```bash
python cli.py load /path/to/docs/directory
```

**Query:**
```bash
python cli.py query "What hardware do I need for farming?"
```

## Example

```
$ python cli.py query "What hardware do I need?"

Query: What hardware do I need?

============================================================
ANSWER
============================================================
The hardware you need depends on whether you're plotting or farming.

For plotting: a fast CPU or GPU, temporary storage (SSD recommended),
and enough RAM. For farming: almost any 64-bit computer made after 2010,
including a Raspberry Pi 4 with 4+ GB RAM.

============================================================
SOURCES
============================================================

[1] docs/reference-client/plotting/plotting-hardware.md
    If you do decide to buy hardware, this page will help you decide...

[2] docs/reference-client/getting-started/farming-guide.md
    Ready? Let's get started! Obtain hardware...
```

## What I'd Build Next

- **Evaluation benchmarks** — measure retrieval accuracy against a test set of question/answer pairs to validate chunk size and embedding choices with data instead of intuition.
- **Multi-format support** — extend beyond markdown to RST, HTML, and PDF so the tool works with any documentation source.
- **Web interface** — a simple frontend where users paste a repo URL and get a searchable docs instance.

## Development

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/ cli.py
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "Rewrite README as case study with architecture and design decisions"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All tests pass.

**Step 2: Run linter**

```bash
ruff check src/ tests/ cli.py
```
Expected: Clean.

**Step 3: Verify CLI**

```bash
python cli.py --help
```
Expected: Clean help output.

**Step 4: Review git log**

```bash
git log --oneline
```
Expected: Clean commit history showing the refactor progression.
