import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("Starting Ingestion Pipeline...")

loader = PyPDFLoader("report.pdf")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(raw_documents)

print(f"Chopped PDF into {len(chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
index_name = "gemini-rag"

PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
print("Success! Gemini data loaded into Pinecone.")