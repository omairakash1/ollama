import faiss
import numpy as np
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.tokenize import PunktTokenizer
import nltk

# Step 1: Load and Preprocess PDF Documents
def load_and_preprocess_pdfs(pdf_paths):
    """
    Loads and preprocesses PDF documents, splitting them into manageable chunks.
    """
    all_documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for path in pdf_paths:
        loader = UnstructuredPDFLoader(path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_documents.extend(chunks)

    return all_documents

# Step 2: Embed and Index Documents
def embed_and_index_documents(documents, embedding_model):
    """
    Embeds text documents and creates a FAISS index for similarity search.
    """
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index, texts

# Step 3: Search FAISS Index
def search_faiss_index(query, index, texts, embedding_model, top_k=5):
    """
    Searches the FAISS index for the most relevant documents based on the query.
    """
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [{"content": texts[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results

# Step 4: Generate Answer Using LLM
from transformers import AutoTokenizer

def generate_answer(query, context, generator):
    """
    Generates an answer using the retrieved context and query.
    """
    # Load tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Construct the prompt
    prompt = f"""
    Context:
    {context}

    Query:
    {query}

    Answer:
    """

    # Tokenize and truncate the input if it exceeds the model's limit
    input_ids = tokenizer(prompt, truncation=True, max_length=900, return_tensors="pt")["input_ids"]

    # Generate response with the remaining token budget
    response = generator(
        tokenizer.decode(input_ids[0], skip_special_tokens=True), 
        max_new_tokens=100,  # Allows for 100 tokens to be generated
        num_return_sequences=1
    )
    return response[0]["generated_text"]


# Step 5: Full Pipeline Integration
def main(pdf_paths, query):
    """
    Integrates all steps into a single pipeline to process PDFs, retrieve relevant data, 
    and generate an answer for the query.
    """
    # Load documents
    print("Loading and preprocessing PDF documents...")
    documents = load_and_preprocess_pdfs(pdf_paths)

    # Initialize embedding model
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Embed and index documents
    print("Embedding and indexing documents...")
    index, texts = embed_and_index_documents(documents, embedding_model)

    # Retrieve relevant documents
    print("Searching FAISS index...")
    results = search_faiss_index(query, index, texts, embedding_model)
    
    # Generate response
    print("Generating response...")
    context = " ".join([result["content"] for result in results])
    generator = pipeline("text-generation", model="gpt2")
    answer = generate_answer(query, context, generator)

    return answer

# Example Usage
if __name__ == "__main__":
    nltk.download('punkt', download_dir='E:/')
    pdf_paths = ["E:/Client Background Information.pdf"]
    query = "What are the key points discussed in the documents?"
    answer = main(pdf_paths, query)
    print("\nGenerated Answer:")
    print(answer)
