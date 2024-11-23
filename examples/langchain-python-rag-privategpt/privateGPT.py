#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time

#ollama.pull("llama3.2:1b")

model = os.environ.get("MODEL", "llama3.2:1b")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS
def initialize_qa_chain(args):
    """
    Initializes the RetrievalQA chain with the provided LLM and embeddings.

    Args:
        args: Command line arguments for controlling output and streaming.

    Returns:
        qa: The RetrievalQA object initialized with the LLM and retriever.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, 
        return_source_documents=not args.hide_source
    )

    return qa

def ask_query(qa, query, hide_source):
    """
    Asks a query to the QA chain and returns the answer and source documents.

    Args:
        qa: The initialized RetrievalQA object.
        query: The input query string.
        hide_source: Boolean flag to hide the source documents.

    Returns:
        tuple: A tuple containing the answer and source documents (if not hidden).
    """
    start = time.time()
    res = qa({"query": query})
    answer, docs = res['result'], [] if hide_source else res['source_documents']
    end = time.time()

    print(f"\nTime taken: {end - start:.2f} seconds")
    return answer, docs

def interactive_qa():
    """
    Interactive session to ask questions to the QA chain.
    """
    args = parse_arguments()
    qa = initialize_qa_chain(args)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query.lower() == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        answer, docs = ask_query(qa, query, args.hide_source)

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def query_qa_function(query, hide_source=False, mute_stream=False):
    """
    Function to ask a single query and get the answer and sources.

    Args:
        query: The input query string.
        hide_source: Boolean flag to hide the source documents.
        mute_stream: Boolean flag to mute the LLM streaming output.

    Returns:
        tuple: A tuple containing the answer and source documents (if not hidden).
    """
    class Args:
        hide_source = hide_source
        mute_stream = mute_stream

    qa = initialize_qa_chain(Args)
    answer, docs = ask_query(qa, query, hide_source)
    return answer, docs

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M", action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    interactive_qa()