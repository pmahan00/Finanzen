from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import argparse
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.llms import Ollama
import ollama
from RAGTestCreateDatabase import format_docs
#llama index part 
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler





CHROMA_PATH = "chroma"



#formatted_prompt = f"Question: {question}\n\nContext: {context}"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
   
    

    #create a retriever 
    retriever = db.as_retriever()
    
 
# Define the Ollama LLM function
    def ollama_llm(query_text, context):
        formatted_prompt = f"Question: {query_text}\n\nContext: {context}"
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
        #print(response)
        return response['message']['content']

# Define the RAG chain
    def rag_chain(query_text):
        retrieved_docs = retriever.invoke(query_text)
        formatted_context = format_docs(retrieved_docs)
        #retrieved documents OK
        return ollama_llm(query_text, formatted_context)
    
    result = rag_chain(query_text)
# Add llama index for true 

    

    

if __name__ == "__main__":
    main()
