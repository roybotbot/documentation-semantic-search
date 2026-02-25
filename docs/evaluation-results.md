# Evaluation results

## Method

20 test queries run against 27 Chia documentation files (435 chunks after splitting). Each query has a category: terminology gap (user language doesn't match doc language), natural language (straightforward "how do I" questions), or hard (deliberately vague or ambiguous).

Two search methods tested on the same queries and the same documents:

- Semantic search: OpenAI text-embedding-3-small embeddings, ChromaDB vector store, top-3 retrieval by cosine distance
- Keyword baseline: TF-IDF with scikit-learn, cosine similarity against full documents, top-3 results

Hits were scored manually by reviewing each result set and deciding whether the returned content actually answered the question. I looked at what came back and judged it.

## Results

|                  | Semantic | Keyword |
|------------------|----------|---------|
| Terminology gap  | 5/7      | 6/7     |
| Natural language | 7/8      | 7/8     |
| Hard             | 5/5      | 2/5     |
| Total            | 17/20    | 15/20   |

Semantic search won overall, but the margin came entirely from the hard queries. On terminology gap and natural language questions, the two methods performed about the same.

## Where semantic search won

The hard queries are where semantic search pulled away. "What should I know before starting," "is it worth it," "how does chia work" — these are vague, don't contain specific keywords, and could plausibly match several documents. Semantic search returned relevant content for all five hard queries. Keyword search got two.

"Is it worth it" is a good example. There's no document with those words in the title. Keyword search returned reference-farming-hardware.md and plotting-hardware.md with very low scores (0.03, 0.02) — basically noise. Semantic search returned pool-farming.md (warning about running into the red), plotting-hardware.md (cost/time tradeoff discussion), and reference-farming-hardware.md (pro-level setups). All three are genuinely relevant to someone asking whether Chia farming is worth the investment.

Semantic search also beat keyword on "how do I install chia on linux" — it returned the farming guide's install section and a FAQ entry about installing from source, both useful. Keyword search's top result was installation.md but the other two results (check-if-things-are-working.md, faq.md) weren't useful enough as a set.

## Where keyword search won

"Can I use a raspberry pi" — Keyword search returned installation.md, which has the Raspberry Pi 4 minimum specs. Semantic search also returned installation.md, but the chunks it found were about swap file configuration and git builds, not the specs section. The relevant content was in the document but chunking surfaced the wrong parts.

"How do I set up a cold wallet for chia" — Keyword search found key-management.md, which has the 2-key cold storage setup instructions. Semantic search returned farming-guide.md and wallet-guide.md install sections. It understood "wallet" but missed "cold" as a qualifier that points to a specific security configuration.

## Where both failed

"How do I get my money" — Neither method connected "money" to "block rewards" or "XCH." Semantic search found FAQ entries about wallet troubleshooting (one even mentions "you don't have enough money in your wallet," but in a debugging context, not an explanation of how rewards work). Keyword search returned plotting-hardware.md, faq.md, and wallet-guide.md — also off-target. The conceptual distance between "get my money" and "block reward distribution" is large enough that neither method bridged it without help.

## What the results tell me

Semantic search earns its keep on vague, underspecified queries where the user can't give you good keywords. That's exactly the use case I built this for — users who know what they want but can't phrase it in the docs' terminology. On those queries, it was 5/5.

But on more specific queries, keyword search keeps up or wins. When the user's words appear literally in the right document ("cold wallet," "raspberry pi"), TF-IDF finds it directly. Semantic search sometimes finds the right document but surfaces the wrong chunk. The content is there but the 1000-character window landed on the wrong paragraph.

The chunking tradeoff showed up repeatedly. Semantic search works on chunks, keyword search worked on full documents. A chunk from the right document can score lower than a chunk from a bigger, more general document (the FAQ came up a lot this way). Keyword search doesn't have this problem because it sees the whole file.

## What I'd change

Bigger chunks would probably help the cases where semantic search found the right document but surfaced the wrong section. 1500 or 2000 characters would keep more context intact per chunk. I'd run the same eval to check whether that actually helps or just shifts which queries fail.

Document-level aggregation after chunk retrieval. If three chunks from the same file all score moderately, that file is probably the right answer even though no single chunk scored highest. The pipeline currently doesn't aggregate by source.

Query expansion for the vocabulary mismatches. "How do I get my money" is the clearest case — expanding it to also search for "rewards," "XCH," "block reward" would likely fix it. Could be an LLM call or a domain-specific synonym map. I'd test both.

A bigger test set. 20 queries shows patterns but isn't enough for real confidence. A change that flips 2-3 queries could be noise. I'd want 50+ queries with at least 15 per category before trusting the numbers.
