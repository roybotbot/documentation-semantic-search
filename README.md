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
git clone https://github.com/roybotbot/chia-docs-semantic-search.git
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
