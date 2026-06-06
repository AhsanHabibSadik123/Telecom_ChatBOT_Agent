"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH = "/media/sadik/Data/my notebook/Gen AI Project/Telecom_ChatBOT_Agent/Resources/telecom_guide.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"{len(pages)} pages loaded.")
    # print(pages[0])
    # print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    for i , chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i
    print(f" {len(chunks)} chunks produce...")

    print("Downloading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. {vectorstore._collection.count()} vectors stored.")

if __name__ == "__main__":
    main()  
