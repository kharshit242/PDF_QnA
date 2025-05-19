from langchain.chains import RetrievalQA
from langchain_nvidia_ai_endpoints import ChatNVIDIA  # Changed import

def get_qa_chain(faiss_index):  # api_key parameter removed
    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1"  # Changed back to ChatNVIDIA and specified model
        # NVIDIA_API_KEY parameter removed, temperature also removed for now
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever())
