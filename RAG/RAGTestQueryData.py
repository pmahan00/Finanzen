from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import argparse
from numpy import result_type
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
# Additional info
import logging
import os
from chromadb.config import Settings
import chromadb
#Streaming 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

#Check if ollm  is working 
ollm = Ollama(
    model="mistral", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)



CHROMA_PATH = "chroma"
print(CHROMA_PATH)


#formatted_prompt = f"Question: {question}\n\nContext: {context}"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    #3 logging 
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    client = chromadb.Client(Settings(anonymized_telemetry=False))


    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #create a retriever 
    retriever = db.as_retriever()
  #  Instantiate the streaming handler callback here 
    streaming_callback_handler = StreamingStdOutCallbackHandler()
  
 
# Define the Ollama LLM function
    def ollama_llm(query_text, context):
        formatted_prompt = f"Question: {query_text}\n\nContext: {context}"
        print(formatted_prompt)
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}],stream=True)
        for chunk in response:
          print(chunk['message']['content'], end='', flush=True)
        return None
        
#Define the RAG chain
    def rag_chain(query_text):
        retrieved_docs = retriever.invoke(query_text)
        formatted_context = format_docs(retrieved_docs)
        #retrieved documents OK
        return ollama_llm(query_text, formatted_context)
    
    result = rag_chain(query_text)
    
    
# Add llama index for true 

    

    

if __name__ == "__main__":
    main()
