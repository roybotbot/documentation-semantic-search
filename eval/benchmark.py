"""Benchmark script comparing semantic search against keyword baseline."""
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path so src/ and eval/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.loader import load_documents, chunk_documents  # noqa: E402
from src.store import create_store  # noqa: E402
from src.query import search_with_scores  # noqa: E402
from eval.keyword_baseline import KeywordSearcher  # noqa: E402


SAMPLE_DATA = str(Path(__file__).parent.parent / "docs" / "sample-data")
RESULTS_DIR = Path(__file__).parent / "results"
QUERIES_FILE = Path(__file__).parent / "test_queries.json"
EVAL_CHROMA_DIR = str(Path(__file__).parent / ".chroma_eval")
REVIEW_FILE = RESULTS_DIR / "review.json"


def index_docs():
    """Load, chunk, and index the sample data. Returns the vector store."""
    if os.path.exists(EVAL_CHROMA_DIR):
        shutil.rmtree(EVAL_CHROMA_DIR)

    print("Indexing sample data...")
    docs = load_documents(SAMPLE_DATA)
    chunks = chunk_documents(docs)
    store = create_store(chunks, persist_dir=EVAL_CHROMA_DIR)
    print(f"Indexed {len(docs)} documents ({len(chunks)} chunks).\n")
    return store


def run_semantic(store, query: str, k: int = 3) -> dict:
    """Run a single query through semantic search."""
    answer, scored_results = search_with_scores(store, query, k=k)
    return {
        "answer": answer,
        "results": [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
                "snippet": doc.page_content[:200],
            }
            for doc, score in scored_results
        ],
    }


def run_keyword(searcher: KeywordSearcher, query: str, k: int = 3) -> dict:
    """Run a single query through keyword search."""
    results = searcher.search(query, k=k)
    return {
        "results": [
            {"source": filename, "score": float(score)}
            for filename, score in results
        ],
    }


def run_benchmark():
    """Run the full benchmark and save results for manual review."""
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    store = index_docs()
    searcher = KeywordSearcher(SAMPLE_DATA)

    results = []

    print("Running queries...\n")

    for q in queries:
        query = q["query"]
        category = q["category"]

        sem_result = run_semantic(store, query)
        kw_result = run_keyword(searcher, query)

        sem_sources = [r["source"].split("/")[-1] for r in sem_result["results"]]
        kw_sources = [r["source"] for r in kw_result["results"]]

        print(f'{q["id"]:>3}. [{category}] {query}')
        print(f"     Semantic: {', '.join(sem_sources)}")
        print(f"     Keyword:  {', '.join(kw_sources)}")
        print()

        results.append({
            "id": q["id"],
            "query": query,
            "category": category,
            "semantic": sem_result,
            "keyword": kw_result,
        })

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RESULTS_DIR / "benchmark_results.json"
    with open(raw_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {raw_file}")

    # Generate review file if it doesn't exist
    if not REVIEW_FILE.exists():
        review = []
        for q in queries:
            review.append({
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "semantic_hit": None,
                "keyword_hit": None,
            })
        with open(REVIEW_FILE, "w") as f:
            json.dump(review, f, indent=2)
        print(f"Review file created at {REVIEW_FILE}")
        print("Fill in semantic_hit and keyword_hit (true/false) then run: python eval/summarize.py")
    else:
        print(f"Review file already exists at {REVIEW_FILE} (not overwritten)")
        print("Run: python eval/summarize.py")


if __name__ == "__main__":
    run_benchmark()
