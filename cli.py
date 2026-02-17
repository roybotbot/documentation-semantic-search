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
