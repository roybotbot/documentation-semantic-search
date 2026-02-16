# Portfolio Refactor Design

## Goal

Turn two working Python scripts into a clean, production-shaped project that signals engineering competence to employers. The Chia docs are just example data — the system works with any markdown documentation. Time budget: one weekend.

## Project Structure

```
chia-docs-semantic-search/
├── src/
│   ├── __init__.py
│   ├── config.py          # All settings: chunk size, model, overlap, k results
│   ├── loader.py           # Document loading and chunking logic
│   ├── store.py            # ChromaDB vector store wrapper
│   └── query.py            # Retrieval and LLM answer generation
├── tests/
│   ├── test_loader.py
│   ├── test_store.py
│   └── test_query.py
├── cli.py                  # Thin entry point, arg parsing only
├── requirements.txt        # Pinned versions
├── .github/
│   └── workflows/
│       └── ci.yml          # Ruff lint + tests on push
├── .gitignore
└── README.md
```

Separate what the code does (src/) from how you run it (cli.py). Every magic number lives in config.py with a comment explaining the choice.

## Code Changes

### Config management
Move all hardcoded values to `config.py`: chunk size (1000), overlap (200), embedding model (`text-embedding-3-small`), LLM model, result count (k=3). Each value gets a comment explaining why it was chosen.

### Error handling
Graceful failures with actionable messages for: missing API key, missing docs directory, empty vector store, API errors. No raw tracebacks.

### Logging
Replace print statements with Python's `logging` module. Separate user-facing output from diagnostic info.

### Deprecation fixes
- Switch from `langchain_community.vectorstores.Chroma` to `langchain_chroma.Chroma`
- Pin all dependency versions in requirements.txt

### Tests
- Unit tests that mock OpenAI calls: test chunking logic, prompt assembly, config values respected
- One optional integration test (needs API key) for a real round-trip
- Use pytest

### CI
GitHub Actions workflow: run `ruff` linter and `pytest` on push. One file, nothing fancy.

## README Structure

1. **One-line summary** — "Turn any documentation repository into a natural language search system using RAG."
2. **Why this exists** — 2-3 sentences. Managed large docs, users struggled with keyword search, built this to solve it.
3. **How it works** — Text-based architecture diagram: Docs → Chunking → Embeddings → Vector Store → Query → LLM → Answer.
4. **Design decisions** — 3-4 short paragraphs on choices and trade-offs: chunk sizes, embedding model, local ChromaDB vs hosted, what worked and what didn't.
5. **Setup & usage** — Short, cleaned up from current README.
6. **Example output** — One trimmed example.
7. **What I'd do next** — 2-3 bullets showing vision: evaluation benchmarks, non-markdown sources, web interface.

## What's NOT in Scope

- No web interface or API server
- No multiple embedding model comparison
- No new features beyond what exists
- No hosted deployment
