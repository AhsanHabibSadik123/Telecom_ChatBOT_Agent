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

