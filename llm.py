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
import argparse
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
PERSISTS_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def load_llm():

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = CTransformers(model='../models/llama-2-7b-chat.ggmlv3.q3_K_L.bin',
                    model_type='llama',
                    callback_manager=callback_manager,
                    config={
                        'gpu_layers' : 15,
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

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # LLM
    # if model_type == "LlamaCpp":
    #     llm = LlamaCpp(model_path=model_path, 
    #                    max_tokens=512, 
    #                    n_batch=model_n_batch, 
    #                    callbacks=callbacks, 
    #                    n_ctx=2048, 
    #                    f16_kv=True,
    #                    n_threads = 4,
    #                    verbose=True)
    # elif model_type == "GPT4All":
    #     llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True)
    # else:
    #     # raise exception if model_type is not supported
    #     raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    llm = load_llm()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ask questions to your documents')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

from langchain.docstore.document import Document

def extract_unique_file_sources(sources):
    """
    extract file source(file name) from documents retreived
    """

    print(sources)

    if 'source_documents' in sources:
        source_documents = [doc for doc in sources['source_documents']]
        unique_sources = set(os.path.basename(doc.metadata['source']) for doc in source_documents)
        return list(unique_sources)
    else:
        return None

def rag_chain(query, persist_directory, top_k = 3):

    args = parse_arguments()

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)

    llama2_llm = load_llm()

    prompt = set_qa_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llama2_llm, 
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

        print(f"\n> Answer (took {round(end - start, 2)} s.):")

        for source in sources:
            print(f"\n> Source: "+ source)

        return answer 
    else:
        raise("Query cannot be None")


if __name__ == "__main__":

    # query = " what is the problems that dialoGPT paper trying to solve?"
    query = " what is dialoGPT?"
    rag_chain(query, PERSISTS_DIRECTORY, top_k=3)