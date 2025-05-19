from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_faiss(docs):  # api_key parameter removed
    # NVIDIA_API_KEY is expected to be in environment variables
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA"
        # NVIDIA_API_KEY parameter removed from constructor
    )
    return FAISS.from_documents(docs, embeddings)
