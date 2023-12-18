from PyPDF2 import PdfReader
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_text_splitter(chunk_size=800, chunk_overlap=200):

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len
    )

    return recursive_text_splitter


def read_one_pdf_file(file_path: str):
    reader = PdfReader(file_path)
    text = ""

    for i, page in enumerate(reader.pages):
        content = page.extract_text()

        if content:
            text += content

    text_splitter = get_text_splitter()

    return text_splitter.split_text(text)


def read_multiple_pdf_files(folder_path : str):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file)
            text = read_one_pdf_file(pdf_path)
            documents.extend(text)

    text_splitter = get_text_splitter()

    chunked_documents = text_splitter.create_documents(documents)

    return chunked_documents

def vector_store(chunked_documents):

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectordb = Chroma.from_documents(chunked_documents, embeddings)

    return vectordb

def load_llm():

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = CTransformers(model='../models/llama-2-7b-chat.ggmlv3.q3_K_L.bin',
                    model_type='llama',
                    callback_manager=callback_manager,
                    config={
                        'max_new_tokens': 256,
                        'temperature': 0.1,
                        'repetition_penalty': 1.18,
                        'context_length' : 2048})
    return llm

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt_template(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def set_qa_prompt():
    sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
    instruction = """CONTEXT:/n/n {context}/n 
    
    Question: {question}"""

    prompt_template = get_prompt_template(instruction, sys_prompt)

    llama_prompt = PromptTemplate(
        template= prompt_template, 
        input_variables=["context", "question"]
        )

    return llama_prompt 


if __name__ == "__main__":

    chunked_documents = read_multiple_pdf_files("documents")
    vector_db = vector_store(chunked_documents)

    llama2_llm = load_llm()

    prompt = set_qa_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llama2_llm, 
        chain_type='stuff',
        retriever=vector_db.as_retriever(search_kwargs={"k":5}),
        verbose=True,
        chain_type_kwargs={
        "verbose": True,
        "prompt": prompt
        })

    query = "what problem that dialoGPT paper trying to solve?"

    start = time.time()

    llm_response = qa_chain(query)

    end = time.time()

    print(llm_response['result'])

    print(f"\n> Answer (took {round(end - start, 2)} s.):")
