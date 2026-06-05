"""
This script is responsible for ingesting PDF documents, splitting them into manageable chunks, and preparing them for embedding and storage in a vector database. It uses the PyPDFLoader to load PDF files, the RecursiveCharacterTextSplitter to split the text into chunks, and the RagChain to create a knowledge graph for efficient retrieval during chatbot interactions.
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = "error"

