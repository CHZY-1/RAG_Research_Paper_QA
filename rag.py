from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
import chromadb
import os
import csv
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

from constants import CHROMA_SETTINGS

EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME")
PERSISTS_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

def load_llm():

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = CTransformers(model='../models/llama-2-7b-chat.ggmlv3.q3_K_L.bin',
                    model_type='llama',
                    callback_manager=callback_manager,
                    config={
                        'gpu_layers' : 30,
                        'max_new_tokens': 512,
                        'temperature': 0.1,
                        'repetition_penalty': 1.18,
                        'context_length' : 2048})
    return llm

def get_prompt_template(instruction, new_system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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


def extract_unique_file_sources(sources):
    """
    extract file source(file name) from documents retreived
    """

    if 'source_documents' in sources:
        source_documents = [doc for doc in sources['source_documents']]
        unique_sources = set(os.path.basename(doc.metadata['source']) for doc in source_documents)
        return list(unique_sources)
    else:
        return None


def write_data_to_csv(data: dict, csv_file_name: str):

    file_exists = os.path.exists(csv_file_name)

    with open(csv_file_name, 'a', newline='') as csv_file:
        fieldnames = ['query', 'answer', 'sources', 'top-k', 'response_time(s)', 'response_length', 'llm', 'llm_type']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)


def rag_chain(query, top_k = 3):

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=PERSISTS_DIRECTORY)
    vector_db = Chroma(persist_directory=PERSISTS_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)

    llm = load_llm()

    prompt = set_qa_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_db.as_retriever(search_kwargs={"k":top_k}),
        return_source_documents= True,
        verbose=False,
        chain_type_kwargs={
        "verbose": False,
        "prompt": prompt
        })
    
    if query:
        start = time.time()

        llm_response = qa_chain(query)
        answer, sources = llm_response['result'], extract_unique_file_sources(llm_response)

        end = time.time()

        response_time = round(end - start, 2)

        print(f"\n> Answer (took {response_time} s.):")

        for source in sources:
            print(f"\n> Source: "+ source)


        data = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'top-k': top_k,
            'response_time(s)': response_time,
            'response_length': len(answer),
            'llm': os.path.basename(llm.model),
            'llm_type': llm.model_type
        }

        write_data_to_csv(data, csv_file_name="rag_research_paper.csv")

        return answer 
    else:
        raise("Query cannot be None")


if __name__ == "__main__":

    query_list = [
        "What data DialoGPT trained on?", 
        "What is the problems that DialoGPT paper trying to solve?", 
        "What is DialoGPT?",
        "How is the architecture of DialoGPT designed to handle conversational context?",
        "Can you explain the key architectural components of DialoGPT mentioned in the paper?",
        "What evaluation metrics were used in the paper to assess the performance of DialoGPT?",
        "Can you discuss the results of the human evaluation mentioned in the paper?",
        "How does DialoGPT compare with other conversation models in terms of performance, according to the paper?",
        "What challenges or limitations does the paper acknowledge in the performance of DialoGPT?",
        "Does the paper discuss any fine-tuning strategies or adaptation techniques for DialoGPT?"
        ]

    for query in query_list:
        rag_chain(query, top_k=3)