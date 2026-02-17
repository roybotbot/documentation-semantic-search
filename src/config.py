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
