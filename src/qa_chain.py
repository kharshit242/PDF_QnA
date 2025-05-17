from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def get_qa_chain(faiss_index, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.2,
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever())
