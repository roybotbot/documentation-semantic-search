# Chia Docs Semantic Search Proof of Concept

Natural language search system for technical documentation using retrieval-augmented generation. Built to demonstrate semantic search implementation on documentation I maintained as Director of Ecosystem Operations at Chia Network.

## Technical Overview
This project implements semantic search over the Chia Network developer documentation repository using vector embeddings and similarity retrieval. The system converts markdown documentation into searchable embeddings, enabling natural language queries that return conceptually relevant results rather than requiring exact keyword matches.
The implementation addresses a common problem in technical documentation: users know what they want to accomplish but don't know the specific terminology used in the docs. Semantic search bridges this gap by understanding intent rather than matching strings.

## Architecture
The system uses retrieval-augmented generation (RAG) architecture with the following components:

- Stack: Python, LangChain, ChromaDB, OpenAI embeddings API

- Document Processing: DirectoryLoader reads markdown files from a locally cloned Chia docs repository. TextSplitter chunks documents into 1000-character segments with 200-character overlap to maintain context across boundaries.

- Vector Storage: ChromaDB stores document embeddings locally. Each chunk gets converted to a 1536-dimension vector using OpenAI's text-embedding-3-small model.

- Retrieval: User queries convert to embeddings using the same model. ChromaDB performs cosine similarity search to return the most relevant document chunks.

## Setup

Use Python 3.8 to 3.13. This does not work on Python 3.14 and above. 

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt --break-system-packages`
3. Clone the Chia Docs repository: `git clone https://github.com/Chia-Network/chia-docs.git`
4. Set OpenAI API key: `export OPENAI_API_KEY='your-key-here'`
5. Load and process documentation: `python load_docs.py`
6. Query the documentation: `python query.py "your question here"`

```bash
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'
python query.py "What hardware do I need for farming?"
```

The `load_docs.py` script clones the Chia documentation repository and processes approximately 200 markdown files into the vector database. This takes 3-5 minutes on first run. Subsequent queries use the cached database.
