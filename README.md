# Documentation semantic search

Semantic search over markdown documentation using RAG. Built from experience maintaining 200+ pages of developer docs at Chia Network.

## The problem

While working on the Chia docs, I kept seeing the same thing: users searched the docs, found nothing, then asked in Discord. The answers were already there. The problem wasn't missing content — it was that users describe things in their own words and the docs use project-specific terminology. Someone searching "how much space do I need" won't find a page about "K32 plot files" and "108GB."

I didn't build a solution for this while at Chia. This project is what I would have built.

## What's in here

A RAG pipeline (load markdown, chunk, embed with OpenAI, store in ChromaDB, retrieve by similarity, answer with an LLM) and an evaluation framework that tests it against a keyword baseline.

The evaluation is the interesting part.

## Evaluation

I wrote 20 test queries based on the kinds of questions users commonly asked, ran them against both semantic search and TF-IDF keyword search on the same 27-doc subset, and scored each result set manually.

|                  | Semantic | Keyword |
|------------------|----------|---------|
| Terminology gap  | 5/7      | 6/7     |
| Natural language | 7/8      | 7/8     |
| Hard queries     | 5/5      | 2/5     |
| Total            | 17/20    | 15/20   |

Semantic search won, but not where I expected. On terminology gap and natural language queries, the two methods performed about the same. The gap came from hard queries — vague, underspecified questions like "is it worth it" and "how does chia work" where keyword search has almost nothing to match on. Semantic search got all five. Keyword search got two.

Keyword search still won on queries where the user's words appear literally in the right document. "Cold wallet" is in the key-management page. "Raspberry pi" is in the installation page. TF-IDF finds those directly. Semantic search sometimes found the right document but surfaced the wrong chunk.

The full analysis is in [evaluation results](docs/evaluation-results.md), and the reasoning behind each technical choice is in [design decisions](docs/design-decisions.md).

## What I'd build next

These come from specific failures in the evaluation.

Document-level aggregation after chunk retrieval. Multiple failures happened because several chunks from the right document each scored moderately, but no single chunk scored highest. The pipeline doesn't know two chunks came from the same file.

Query expansion for vocabulary mismatches. "How do I get my money" was the clearest failure — neither method connected "money" to "block rewards" or "XCH." Expanding queries into related terms before searching would likely fix this. Either an LLM generates the variants or a domain-specific synonym map handles it.

Larger chunks. Several cases where semantic search found the right document but the 1000-character window landed on the wrong paragraph. 1500 or 2000 characters might fix that, though it could create different problems. I'd want to test it against the same evaluation set.

## How it works

```
Markdown files
    │
    ▼
┌──────────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────────┐
│ Preprocessor │───▶│  Loader  │───▶│   Chunker    │───▶│  Embeddings │
│ (clean MDX)  │    │ (md/mdx) │    │ (1000 char)  │    │  (OpenAI)   │
└──────────────┘    └──────────┘    └──────────────┘    └──────┬──────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │  ChromaDB   │
                                                        │ (local)     │
                                                        └──────┬──────┘
                                                               │
User query ──▶ Embed ──▶ Similarity Search ────────────────────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │   LLM       │──▶ Answer + Sources
                                                        └─────────────┘
```

The preprocessor strips YAML frontmatter, MDX/JSX components, import statements, and Docusaurus admonition syntax before chunking. Without this, component tags and metadata pollute the embeddings.

## Setup

Requires Python 3.8-3.13 and an OpenAI API key.

```bash
git clone https://github.com/roybotbot/chia-docs-semantic-search.git
cd chia-docs-semantic-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Usage

Index the sample data:
```bash
python cli.py load docs/sample-data
```

Query:
```bash
python cli.py query "What hardware do I need for farming?"
```

Show retrieval scores and chunk details:
```bash
python cli.py query --explain "What hardware do I need for farming?"
```

Run the evaluation:
```bash
python eval/benchmark.py
```

## Development

```bash
pytest tests/ -v
ruff check src/ tests/ cli.py eval/
```
