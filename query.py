import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise ValueError("OPENAI_API_KEY environment variable not set")
	
# Get query from command line
if len(sys.argv) < 2:
	print("Usage: python query.py 'your question here'")
	sys.exit(1)
	
query = " ".join(sys.argv[1:])

print(f"Query: {query}\n")

# Initializing embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Loading vector store...")
vectorstore = Chroma(
	persist_directory="./chroma_db",
	embedding_function=embeddings
)

print("Searching documentation...\n")
results = vectorstore.similarity_search_with_score(query, k=3)

print("    ᕕ( ՞ ᗜ ՞ )ᕗ    =\n" * 4)
print("RESULTS")
print("	   ᕕ( ՞ ᗜ ՞ )ᕗ    =\n" * 4)

for i, (doc, score) in enumerate(results, 1):
	print(f"\n[Result {i}] Relevance Score: {score:.4f}")
	print(f"Source: {doc.metadata.get('source', 'Unknown')}")
	print(f"\nContent:\n{doc.page_content}")
	print("-" * 80)

print(f"\nReturned {len(results)} results")