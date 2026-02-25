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


def search_with_scores(store, query: str, k: int = config.RETRIEVAL_K) -> tuple:
    """Search with similarity scores for retrieval debugging.

    Args:
        store: Chroma vector store instance.
        query: The user's question.
        k: Number of similar documents to retrieve.

    Returns:
        Tuple of (answer_text, list of (document, score) pairs).
    """
    logger.info("Searching (with scores) for: %s", query)
    results = store.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _score in results]
    logger.info("Retrieved %d documents", len(docs))

    prompt = build_prompt(query, docs)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    response = llm.invoke(prompt)

    return response.content, results
