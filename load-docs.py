import os
import subprocess
import shutil
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

print("Cloning Chia documentation from GitHub...")
repo_path = Path("./temp_chia_docs")

# Remove if exists
if repo_path.exists():
	shutil.rmtree(repo_path)

# Clone repo
subprocess.run([
	"git", "clone", 
	"--depth", "1",
	"https://github.com/chia-network/chia-docs.git",
	str(repo_path)
], check=True)

print("Loading documentation files...")
loader = DirectoryLoader(
	str(repo_path / "docs"),
	glob="**/*.md",
	loader_cls=UnstructuredMarkdownLoader
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Clean up cloned repo
print("Cleaning up temporary files...")
shutil.rmtree(repo_path)

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
print(f"Total documents processed: {len(docs)}")
print(f"Total chunks created: {len(all_splits)}")
print(f"Vector store saved to: ./chroma_db")

# Show sample chunk
if all_splits:
	print(f"\nSample chunk (first 200 chars):")
	print(all_splits[0].page_content[:200])