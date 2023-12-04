from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

def extract_txt_from_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        txt = ""
        for page in pdf_reader.pages:
            txt += page.extract_text()

        return txt

def run_app():
    load_dotenv()
    st.set_page_config(page_title="Paper QA Chatbot")
    st.header("Chatbot")

    document_pdf = st.file_uploader("Upload PDF", type="pdf")
    txt = extract_txt_from_pdf(document_pdf)
    st.write(txt)

if __name__ == "__main__":
    run_app()