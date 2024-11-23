#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain import PromptTemplate
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "llama3.2:1b")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    
    # Initialize the embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Set up retriever with search configuration
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Initialize LLM model (Ollama)
    llm = Ollama(model=model, callbacks=callbacks)

    # Define a template for the QA prompt
    template = """
    Given the context below, answer the question at the end as concisely and coherent as possible. If unsure, simply say "I don't know".
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

    # Create the QA chain with the customized prompt
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source, chain_type_kwargs={"prompt": prompt_template})

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain using the query and context
        start = time.time()
        res = qa({"query": query})
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ask questions to your documents using LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Disable printing of source documents.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Disable the streaming callback for LLMs.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
