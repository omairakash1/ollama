#!/usr/bin/env python3
import os
import glob
import logging
import pickle
import faiss
import torch
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "db")
SOURCE_DIRECTORY = os.environ.get("SOURCE_DIRECTORY", "source_documents")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Map file extensions to loaders
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyMuPDFLoader,
    # Extend this mapping with custom loaders for other file types
}

def load_documents(source_dir: str) -> List[str]:
    """Load all documents from the source directory."""
    file_paths = []
    for ext in LOADER_MAPPING.keys():
        file_paths.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    documents = []
    for file_path in tqdm(file_paths, desc="Loading documents"):
        _, ext = os.path.splitext(file_path)
        loader_class = LOADER_MAPPING.get(ext)
        if loader_class:
            try:
                loader = loader_class(file_path)
                documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
    return documents

def split_text(documents: List[str], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Split documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(model_name: str):
    """Load a Hugging Face model and tokenizer for embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def embed_texts(texts: List[str], tokenizer, model, batch_size=16) -> torch.Tensor:
    """Generate embeddings for a list of texts."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings, dim=0)

def save_vectorstore(vectors: torch.Tensor, metadata: List[Dict], persist_directory: str):
    """Save vectors and metadata using FAISS."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Save vectors using FAISS
    index = faiss.IndexFlatL2(vectors.size(1))
    index.add(vectors.numpy())
    faiss.write_index(index, os.path.join(persist_directory, "index.faiss"))
    
    # Save metadata as a pickle
    with open(os.path.join(persist_directory, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def load_vectorstore(persist_directory: str):
    """Load vectors and metadata."""
    index = faiss.read_index(os.path.join(persist_directory, "index.faiss"))
    with open(os.path.join(persist_directory, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def main():
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        logging.info("No documents found.")
        return

    # Split documents into chunks
    chunks = split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_texts = [chunk.page_content for chunk in chunks]
    metadata = [{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks]
    logging.info(f"Generated {len(chunk_texts)} text chunks.")

    # Create embeddings
    tokenizer, model = create_embeddings(EMBEDDINGS_MODEL_NAME)
    vectors = embed_texts(chunk_texts, tokenizer, model)

    # Save vectorstore
    save_vectorstore(vectors, metadata, PERSIST_DIRECTORY)
    logging.info("Vectorstore updated successfully.")

if __name__ == "__main__":
    main()
