import os
import sys
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise ValueError("OPENAI_API_KEY environment variable not set")

# Get query from command line
if len(sys.argv) < 2:
	print("Usage: python query.py 'your question here'")
	sys.exit(1)

query = " ".join(sys.argv[1:])

print(f"Query: {query}\n")

# Initialize embeddings (must match what was used in load_docs.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize chat model for answer generation
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# Load the persisted vector store
print("Loading vector store...")
vectorstore = Chroma(
	persist_directory="./chroma_db",
	embedding_function=embeddings
)

# Search for similar documents
print("Retrieving relevant documentation...\n")
retrieved_docs = vectorstore.similarity_search(query, k=3)

# Format retrieved context for the model
docs_content = "\n\n".join(
	f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
	for doc in retrieved_docs
)

# Create prompt with context
prompt = f"""You are a helpful assistant answering questions about the Chia blockchain documentation.

Use the following context to answer the query. If the answer cannot be found in the context, say so.

Context:
{docs_content}

Query: {query}

Answer:"""

# Generate answer using the chat model
print("Generating answer...\n")
response = llm.invoke(prompt)

# Display results
print("=*=" * 20)
print("ANSWER")
print("=*=" * 20)
print(response.content)
print("\n" + "=*=" * 20)
print("SOURCES")
print("=*=" * 20)

for i, doc in enumerate(retrieved_docs, 1):
	print(f"\n[Source {i}]")
	print(f"File: {doc.metadata.get('source', 'Unknown')}")
	print(f"Content preview: {doc.page_content[:200]}...")
	print("-" * 80)