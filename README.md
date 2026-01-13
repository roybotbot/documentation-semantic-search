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

## Example Output

### Command
```bash

➜  chia-docs-semantic-search git:(main) ✗ python query.py "What hardware do I need?"
```

### Output
```bash
Query: What hardware do I need?

Loading vector store...
/Users/roy/Projects/chia-docs-semantic-search/query.py:28: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
  vectorstore = Chroma(
Retrieving relevant documentation...

Generating answer...

=*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*=
ANSWER
=*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*=
Answer (based on the docs):

- The hardware you need depends on whether you’re plotting or farming.

Plotting hardware (to create plots)
- Main components: temporary storage and a processor (CPU or GPU).
- Temporary storage tradeoffs:
  - RAM: fastest and doesn’t wear out from plotting, but requires a high-end workstation; tends to be economical mainly for large farms (> ~1 PiB).
  - HDD: inexpensive and durable, but significantly slower.
  - SSD: fast and practical; SSDs can wear out over time, so high-endurance enterprise NVMe is recommended.
- In short: more compute and faster temporary storage speed mean faster plotting. RAM, SSDs, and/or GPUs/CPUs are the key choices. Most setups use a combination that fits budget and plot rate needs.

Farming hardware (to store and farm plots)
- You need a 64-bit CPU (and a 64-bit OS) since farming is done on a 64-bit platform.
- Windows, Linux, and macOS are supported.
- Minimum for farming (from the farming guide): a Raspberry Pi 4 with 4 GB RAM for a CLI farm, or 8 GB RAM for a GUI farm.
- Most computers made after 2010 can be used for farming.
- Plotting is resource-intensive, and while a Pi can be used for plotting (it will be slow), it’s not ideal long-term.

Additional note
- The page mentions a new proof format introduced in 2024, which will have slightly different hardware requirements for plotting and farming; the remainder of the page describes the original format.

=*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*=
SOURCES
=*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*==*=

[Source 1]
File: /Users/roy/Projects/chia-docs/docs/reference-client/plotting/plotting-hardware.md
Content preview: If you do decide to buy hardware, this page will help you to decide what might work best for your farm.

When looking for a plotting machine, the main components to consider are the temporary storage ...
--------------------------------------------------------------------------------

[Source 2]
File: /Users/roy/Projects/chia-docs/docs/reference-client/getting-started/farming-guide.md
Content preview: :::

Ready? Let's get started!

Obtain hardware

You may already have everything you need, but let's make sure. (All you need for this tutorial is the minimum requirements. We'll cover more optimized ...
--------------------------------------------------------------------------------

[Source 3]
File: /Users/roy/Projects/chia-docs/docs/reference-client/plotting/plotting-hardware.md
Content preview: sidebar_label: Hardware title: Plotting Hardware slug: /reference-client/plotting/plotting-hardware

import Tabs from '@theme/Tabs'; import TabItem from '@theme/TabItem';

New proof format

In 2024, w...
```
