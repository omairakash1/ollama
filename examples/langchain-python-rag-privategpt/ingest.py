#!/usr/bin/env python3
import os
import logging
import pickle
import faiss
import torch
import glob  # Import glob for file path matching
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredMarkdownLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
chunk_size = 500
chunk_overlap = 50

# Document loader mapping
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyMuPDFLoader,
}

def load_documents(source_dir: str) -> List[str]:
    """Load all documents using LangChain loaders."""
    documents = []
    for ext, loader_class in LOADER_MAPPING.items():
        file_paths = glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        for file_path in file_paths:
            try:
                loader = loader_class(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
    return documents

def split_documents(documents: List[str]) -> List[Dict]:
    """Split documents into chunks using LangChain's text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadata = []
    for i, doc in enumerate(documents):
        doc_chunks = text_splitter.split_text(doc.page_content)
        chunks.extend(doc_chunks)
        metadata.extend([{"doc_id": i, "chunk_id": j, "source": doc.metadata} for j in range(len(doc_chunks))])
    return chunks, metadata

def create_embeddings(model_name: str):
    """Load Hugging Face model and tokenizer for embedding generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def embed_texts(texts: List[str], tokenizer, model, batch_size=16) -> torch.Tensor:
    """Generate embeddings for a list of texts using Hugging Face."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings, dim=0)

def save_vectorstore(vectors: torch.Tensor, metadata: List[Dict], persist_directory: str):
    """Save FAISS index and metadata."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Save vectors using FAISS
    index = faiss.IndexFlatL2(vectors.size(1))
    index.add(vectors.numpy())
    faiss.write_index(index, os.path.join(persist_directory, "index.faiss"))
    
    # Save metadata as a pickle file
    with open(os.path.join(persist_directory, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def load_vectorstore(persist_directory: str):
    """Load FAISS index and metadata."""
    index = faiss.read_index(os.path.join(persist_directory, "index.faiss"))
    with open(os.path.join(persist_directory, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def main():
    logging.info(f"Loading documents from {source_directory}")
    
    # Load and split documents
    raw_documents = load_documents(source_directory)
    if not raw_documents:
        logging.info("No documents found.")
        return
    
    logging.info(f"Loaded {len(raw_documents)} documents.")
    chunks, metadata = split_documents(raw_documents)
    logging.info(f"Generated {len(chunks)} text chunks.")

    # Create embeddings
    tokenizer, model = create_embeddings(embeddings_model_name)
    vectors = embed_texts(chunks, tokenizer, model)

    # Save vectorstore
    save_vectorstore(vectors, metadata, persist_directory)
    logging.info("Vectorstore updated successfully.")

if __name__ == "__main__":
    main()
