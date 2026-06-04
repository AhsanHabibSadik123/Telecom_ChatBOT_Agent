from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import sqlite3

CHROMA_DIR = "chroma_store"
COLLECTION = "tickets"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TICKET_DIR = "E:\\Agentic_AI\\Telecom_ChatBOT\\Resources\\tickets.db"

def load_ticket_documents(ticket_dir):
    conn = sqlite3.connect(TICKET_DIR)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM tickets where status = 'resolved'").fetchall()
    docs = []
    for row in rows:
        id, ticket_id, category, issue_type, description, resulation, status = row
        content = (
            f"Issue: {issue_type}\n"
            f"Description: {description}\n"
            f"Resulation: {resulation}\n"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source" : "ticket",
                "ticket_id" : ticket_id,
                "category" : category,
                "status" : status
            }
        ))
    return docs









