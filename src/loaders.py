from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str, chunk_size=1000, overlap=200):
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(pages)
