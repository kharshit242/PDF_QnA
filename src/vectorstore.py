from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Updated import
from dotenv import load_dotenv # Added import

load_dotenv() # Added call to load environment variables

def build_faiss(docs, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    return FAISS.from_documents(docs, embeddings)
