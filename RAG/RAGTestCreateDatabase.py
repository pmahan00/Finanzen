#from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain.embeddings import HuggingFaceEmbeddings# Import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import progressbar
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os

CHROMA_PATH = "chroma"
DATA_PATH = "test/testdata"


# Assuming the current working directory is /home/tonoi/FinAdv/test
# and you want to make DATA_PATH relative to this directory.

# Current absolute path
current_directory = os.getcwd()

# Convert DATA_PATH to a relative path
relative_data_path = os.path.relpath(DATA_PATH, current_directory)

print(relative_data_path)


def main():
    generate_data_store()

def generate_data_store():
    docs = load_docs()
    chunks = split_text(docs)
    save_to_chroma(chunks)

def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docs = loader.load()
    return docs

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    #print(f"Split {len(docs)} docs into {len(chunks)} chunks.")

    #document = chunks[10]
    #print(document.page_content)
    #print(document.metadata)

    return chunks

def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)



def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize HuggingFaceEmbeddings with the 'nlp/gte-small' model
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # Create a new DB from the docs.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    
    # Add a progress bar using progressbar2
    
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()