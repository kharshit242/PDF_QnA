from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str, chunk_size=1000, overlap=200):
    loader = PyMuPDFLoader(pdf_path)
    pages  = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(pages)
