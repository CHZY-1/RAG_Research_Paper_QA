from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from documents_embedding import get_vector_store, write_data_to_csv
from rag_llm import load_llm
import os
import time
from constants import load_environment

load_environment()
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME")
PERSISTS_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

from langchain.callbacks.base import BaseCallbackHandler

def get_stream_handler(chat_box):
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text) 

    return StreamHandler(chat_box)
    

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
    
    Question: {question}""".strip(" ")

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

def get_rag_chain(model='llama2_7b', top_k = 3):
    vector_db = get_vector_store()

    llm = load_llm(model)

    prompt = set_qa_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_db.as_retriever(search_kwargs={"k":top_k}),
        return_source_documents= True,
        chain_type_kwargs={
        "verbose": False,
        "prompt": prompt
        })
    
    return qa_chain, llm

import asyncio
async def get_rag_chain_async(model='llama2_7b', top_k=3):
    vector_db = get_vector_store()

    llm = load_llm(model)

    prompt = set_qa_prompt()

    loop = asyncio.get_event_loop()
    qa_chain = await loop.run_in_executor(None, lambda: RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_db.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt
        }
    ))

    return qa_chain


def rag_chain(query, model='llama2_7b', top_k=3):

    qa_chain, llm = get_rag_chain(model, top_k=top_k)
    
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

        # fieldnames = ['query', 'answer', 'sources', 'top-k', 'response_time(s)', 'response_length', 'llm', 'llm_type']

        write_data_to_csv(data, csv_file_name="rag_research_paper.csv")

        return answer 
    else:
        raise("Query cannot be None")


if __name__ == "__main__":

    # dialogpt_question_list = [
    #     "What data DialoGPT trained on?", 
    #     "What is the problems that DialoGPT paper trying to solve?", 
    #     "What is DialoGPT?",
    #     "How is the architecture of DialoGPT designed to handle conversational context?",
    #     "Can you explain the key architectural components of DialoGPT mentioned in the paper?",
    #     "What evaluation metrics were used in the paper to assess the performance of DialoGPT?",
    #     "Can you discuss the results of the human evaluation mentioned in the paper?",
    #     "How does DialoGPT compare with other conversation models in terms of performance, according to the paper?",
    #     "What challenges or limitations does the paper acknowledge in the performance of DialoGPT?",
    #     "Does the paper discuss any fine-tuning strategies or adaptation techniques for DialoGPT?"
    #     ]

    transformer_question_list = [
        "What is the fundamental concept introduced in the 'Attention is All You Need' paper?",
        "How does the transformer architecture in the paper handle sequential data, and what advantages does it offer over traditional sequence-to-sequence models?",
        # "What specific problems or challenges does the 'Attention is All You Need' paper aim to address in the context of neural machine translation?",
        # "Can you provide insights into the self-attention mechanism introduced in the paper and how it contributes to the model's ability to capture long-range dependencies in sequences?",
        # "In the 'Attention is All You Need' paper, what are the key components of the transformer model architecture, and how do they work together to process input sequences?",
        # "How is positional encoding incorporated into the transformer model, as described in the paper, to account for the sequential nature of input data?",
        # "What are the quantitative evaluation metrics used in the paper to assess the performance of the transformer model, and how do they contribute to understanding the model's capabilities?",
        # "Can you discuss the empirical results presented in the 'Attention is All You Need' paper, highlighting key findings related to the model's performance on different tasks or datasets?",
        # "According to the paper, how does the transformer model compare with earlier models in terms of computational efficiency and parallelization capabilities?",
        # "Does the 'Attention is All You Need' paper discuss any potential limitations or challenges associated with the transformer architecture, and what future directions are suggested for further improvement?"
    ]

    # for query in transformer_question_list:
    #     rag_chain(query, top_k=3)

    # rag_chain("What is K, Q, V and how their are used in attention mechanism", top_k=3)

    query = "Does Ragas paper involve using LLM to evaluate another LLM?"
    open_query = "Tell me highest mountain in the world?"
    rag_chain(open_query, top_k=3)