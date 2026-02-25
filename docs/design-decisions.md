# Design decisions

## Chunk size: 1000 characters, 200 overlap

I went with LangChain's commonly recommended defaults and didn't test alternatives. 1000 characters is roughly a paragraph or two — enough to hold a self-contained idea, small enough that a retrieval hit points to something specific. The 200-character overlap prevents sentences from getting split across chunk boundaries.

Looking at the evaluation results now, I think this was a mistake. Not the choice itself, but the fact that I didn't experiment. Multiple retrieval failures traced back to chunking: the right document's chunks scored individually lower than chunks from bigger, more general documents like the FAQ. Larger chunks (1500-2000) might fix that. Or they might just create different problems. I don't know, because I didn't test it.

This is the first thing I'd change.

## Embedding model: text-embedding-3-small

OpenAI's cheapest embedding model, 1536 dimensions. For 27 documents the cost difference between models is negligible, so this was a default rather than a decision.

I don't think a better embedding model would change the results much. The failures I saw in the evaluation were concept problems. "How do I get my money" failing to find a page about "block rewards" isn't a vector distance issue. The embedding model understands both phrases fine individually. The gap is that the user's mental model ("I want money") and the docs' framing ("block reward distribution in XCH") are just far apart.

## ChromaDB, locally

27 files, 435 chunks. A hosted vector database would add authentication, network calls, and infrastructure management for zero benefit. ChromaDB runs as a local SQLite-backed store and indexes in seconds.

If this were a real production system with thousands of documents and concurrent users, I'd pick something hosted. Adding infrastructure to a portfolio project just to have infrastructure would be dishonest about the actual requirements.

## LangChain

LangChain handles document loading, text splitting, embedding, and retrieval. I could have had each piece written from scratch with the OpenAI SDK and ChromaDB client directly.

I didn't because the plumbing isn't interesting here. The evaluation methodology is. Having my own text splitter would add 50 lines of code and zero insight. LangChain's a heavy dependency, and in a project where I needed fine control over the retrieval pipeline, it would get in the way. For this scope it's fine.

## Preprocessing

The Chia docs are MDX files. They have YAML frontmatter, JSX component tags (`<Tabs>`, `<TabItem>`), ES-style import statements, Docusaurus admonition syntax. None of that is searchable content. Left in, it pollutes the embeddings. A chunk that's half JSX tags is carrying noise that pushes it away from relevant queries in vector space.

The preprocessor is regex-based and handles the specific patterns that appear in the Chia docs. A proper solution would parse the MDX AST. I didn't go that far because the regex approach handles everything in the sample data and I'd rather spend the complexity budget on evaluation.

One thing I should have done: extract the frontmatter title and slug as metadata before stripping them. If a user searches "farming guide" and there's a document literally titled "Farming Guide," matching on that title field would be more precise than hoping the right chunk floats to the top by body text alone.

## TF-IDF keyword baseline

The baseline is deliberately simplistic. TF-IDF vectorization, cosine similarity, full documents (not chunks). No stemming, no BM25, no fuzzy matching.

I could have used BM25, which generally outperforms TF-IDF on retrieval benchmarks. I didn't because the baseline represents what users actually get from a typical docs site search — basic keyword matching. The comparison I wanted was "semantic search vs. the naive alternative," not "semantic search vs. an optimized keyword system." Making the baseline smarter would have muddied that.

And then it beat semantic search anyway, which was more useful than a clean win would have been.

## k=3 retrieval

Three results per query. I didn't test k=5 or k=10. Higher k mechanically improves hit rates (more chances to land on the right doc) but adds noise to the LLM prompt. For a search results page you'd show more. For a RAG prompt you want fewer. Three felt like a fair test.

## What I didn't build, and why

Re-ranking with a cross-encoder after initial retrieval would probably fix several of the evaluation failures. The bi-encoder embedding retrieval is fast but coarse — it misses cases where the query and the relevant chunk share meaning but not vocabulary. A cross-encoder scores the pair jointly and catches more. I didn't build it because I wanted to see how the base pipeline performed first. Can't improve what you haven't measured.

Query expansion would address the vocabulary mismatch problem directly. Expand "how much space do I need" into ["plot size", "K32", "storage requirements"] before searching. Could be an LLM call or a hand-built synonym map. I'd want to test both against the eval set before picking one, and I hadn't built the eval set yet when I was making these decisions.

Incremental indexing (only re-index changed files) would matter in production but is irrelevant for an evaluation project with 27 static files.

Metadata filtering (scoping queries to document categories) would help at scale. A troubleshooting query doesn't need results from the architecture overview. With 27 documents it doesn't matter enough to build.
