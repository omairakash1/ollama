#!/usr/bin/env python3
import os
import argparse
import time
import faiss
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

# Environment variables and constants
model_name = os.environ.get("MODEL", "gpt2")  # Replace with desired Hugging Face model
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

def load_vectorstore(persist_directory: str):
    """Load FAISS index and metadata."""
    index_path = os.path.join(persist_directory, "index.faiss")
    metadata_path = os.path.join(persist_directory, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise ValueError("Persisted vectorstore not found. Please build it first.")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def query_vectorstore(index, metadata, query_embedding, k=4):
    """Query FAISS index for nearest neighbors."""
    distances, indices = index.search(query_embedding, k)
    results = [{"content": metadata[i]["content"], "source": metadata[i]["source"]} for i in indices[0] if i != -1]
    return results

def generate_embeddings(texts, tokenizer, model, batch_size=16):
    """Generate embeddings using Hugging Face models."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)

def main():
    # Parse arguments
    args = parse_arguments()

    # Load vectorstore
    index, metadata = load_vectorstore(persist_directory)

    # Load Hugging Face model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
    embeddings_model = AutoModelForCausalLM.from_pretrained(embeddings_model_name)
    embeddings_model.eval()

    # Load Hugging Face pipeline for text generation
    generator = pipeline("text-generation", model=model_name)

    print("Vectorstore loaded. Ready for queries!")

    # Interactive loop
    while True:
        query = input("\nEnter a query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        # Generate query embedding
        query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            query_embedding = embeddings_model(**query_tokens).last_hidden_state.mean(dim=1).numpy()

        # Query the vectorstore
        results = query_vectorstore(index, metadata, query_embedding, k=target_source_chunks)

        # Combine context from retrieved documents
        context = " ".join([result["content"] for result in results])

        # Generate response
        prompt = f"""
        Given the context below, extract the following details and present them in a single, well-organized paragraph:
        - Store Type
        - Store Offerings
        - Store Size

        Context: {context}
        Answer:
        """
        response = generator(prompt, max_length=300, num_return_sequences=1)[0]["generated_text"]

        # Output response
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(response)

        # Optionally print sources
        if not args.hide_source:
            print("\n> Sources:")
            for result in results:
                print(f"- {result['source']}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ask questions to your documents using LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Disable printing of source documents.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
