from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_faiss(docs):  # api_key parameter removed
    if not docs:
        raise ValueError("No documents provided to build_faiss. Please check your document loading logic.")
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA"
        # NVIDIA_API_KEY parameter removed from constructor
    )
    try:
        return FAISS.from_documents(docs, embeddings)
    except IndexError:
        raise ValueError("Embeddings could not be generated for the provided documents. Please check document content and embedding model.")
