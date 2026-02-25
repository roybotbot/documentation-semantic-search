"""CLI entry point for the semantic search system."""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


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
        print("Error: OPENAI_API_KEY not set.")
        print("Add it to .env or export it: export OPENAI_API_KEY='your-key'")
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
    from src.store import load_store
    from src.query import search_and_answer, search_with_scores

    check_api_key()
    store = load_store()
    explain = getattr(args, "explain", False)

    if explain:
        answer, scored_results = search_with_scores(store, args.question)
        print(f"\nQuery: {args.question}\n")
        print("=" * 60)
        print("RETRIEVAL DETAILS")
        print("=" * 60)
        for i, (doc, score) in enumerate(scored_results, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"\n[{i}] {source}")
            print(f"    Score: {score:.4f}")
            print(f"    {doc.page_content[:150]}...")
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
    else:
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
    query_parser.add_argument(
        "--explain", action="store_true",
        help="Show retrieval scores and chunk details"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "load":
        cmd_load(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
