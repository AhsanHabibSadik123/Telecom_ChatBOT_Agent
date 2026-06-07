"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

def build_context(query: str) -> str:
    docs = build_retriever(query)
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def build_chain():
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\nQuestion: {question}\nAnswer:")
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": RunnableLambda(build_context), "question": RunnableLambda(lambda x: x)}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    response = chain.invoke("My internet is not working, what should I do?")
    print(response)