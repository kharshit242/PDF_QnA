import os, tempfile, streamlit as st
from dotenv import load_dotenv

from src.loaders     import load_and_split
from src.vectorstore import build_faiss
from src.qa_chain    import get_qa_chain

# ----------------- Setup -----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Q&A (Gemini)", page_icon="📄")
st.title("📄 Ask questions about your PDF")

# ----------------- Upload -----------------
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    # Build index (cache in session_state for speed)
    if "qa" not in st.session_state:
        with st.spinner("Indexing…"):
            docs   = load_and_split(pdf_path)
            faiss  = build_faiss(docs, API_KEY)
            st.session_state.qa = get_qa_chain(faiss, API_KEY)

    # ----------------- Q&A -----------------
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Thinking…"):
            answer = st.session_state.qa.run(question)
            st.markdown(f"**Answer:** {answer}")
