import os 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd

CHROMA_DIR = "chroma_store"
COLLECTION = "faq"
CSV_PATH = "E:\\Agentic_AI\\Telecom_ChatBOT\\Resources\\faq.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents(faq_path):
    df = pd.read_csv(CSV_PATH)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        page_content=content,
        metadata={"source": "faq", "category": row['category'], "faq_id": row['id']}
    return docs

def main():
    print("Loading FAQ documents...")
    documents = load_faq_documents(CSV_PATH)
    print(f"Number of FAQ documents loaded: {len(documents)}")

    print("Downloading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Embedding model downloaded.")
    print("Embedding FAQ documents and storing in Chroma...")

    vector_store = Chroma.from_documents(
        documents= documents,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR
    )
    print(f"Done! \n {vector_store._collection.count()} vectors stored in Chroma collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()