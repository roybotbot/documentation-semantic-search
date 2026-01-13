import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise ValueError("OPENAI_API_KEY environment variable not set")

print("Initializing embeddings model...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Get directory path from user
docs_path = input("Enter the path to the chia-docs/docs directory: ").strip()
docs_path = Path(docs_path).expanduser()

if not docs_path.exists():
	raise ValueError(f"Directory does not exist: {docs_path}")

if not docs_path.is_dir():
	raise ValueError(f"Path is not a directory: {docs_path}")

print(f"Loading documentation from: {docs_path}")
print("Loading documentation files (.md and .mdx only)...")

# Load .md files
loader_md = DirectoryLoader(
	str(docs_path),
	glob="**/*.md",
	loader_cls=UnstructuredMarkdownLoader
)
docs_md = loader_md.load()
print(f"Loaded {len(docs_md)} .md files")

# Load .mdx files
loader_mdx = DirectoryLoader(
	str(docs_path),
	glob="**/*.mdx",
	loader_cls=UnstructuredMarkdownLoader
)
docs_mdx = loader_mdx.load()
print(f"Loaded {len(docs_mdx)} .mdx files")

# Combine both
docs = docs_md + docs_mdx
print(f"Total documents loaded: {len(docs)}")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000,
	chunk_overlap=200,
	add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks")

print("Creating vector store and generating embeddings...")
print("(This will take several minutes and use OpenAI API credits)")
vectorstore = Chroma.from_documents(
	documents=all_splits,
	embedding=embeddings,
	persist_directory="./chroma_db"
)

print(f"\nIndexing complete!")
print(f"Total .md files: {len(docs_md)}")
print(f"Total .mdx files: {len(docs_mdx)}")
print(f"Total documents processed: {len(docs)}")
print(f"Total chunks created: {len(all_splits)}")
print(f"Vector store saved to: ./chroma_db")

# Show sample chunk
if all_splits:
	print(f"\nSample chunk (first 200 chars):")
	print(all_splits[0].page_content[:200])