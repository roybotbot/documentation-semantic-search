import os
import subprocess
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

print("Cloning Chia documentation from GitHub...")
repo_path = Path("./temp_chia_docs")

# Remove if exists
if repo_path.exists():
	shutil.rmtree(repo_path)
	
# Clone repo
	subprocess.run([
		"git", "clone", 
		"--depth", "1",  # shallow clone, faster
		"https://github.com/chia-network/chia-docs.git",
		str(repo_path)
	], check=True)
	
# Load and chunk contents from Chia Docs
print("Loading documentation files...")
	loader = DirectoryLoader(
		str(repo_path / "docs"),
		glob="**/*.md",
		loader_cls=UnstructuredMarkdownLoader
	)
	docs = loader.load()
	print(f"Loaded {len(docs)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
	"""Retrieve information to help answer a query."""
	retrieved_docs = vector_store.similarity_search(query, k=2)
	serialized = "\n\n".join(
		(f"Source: {doc.metadata}\nContent: {doc.page_content}")
		for doc in retrieved_docs
	)
	return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
	"You have access to a tool that retrieves context from a blog post. "
	"Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# Clean up cloned repo
print("Cleaning up temporary files...")
shutil.rmtree(repo_path)