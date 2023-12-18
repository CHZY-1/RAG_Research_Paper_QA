import streamlit as st

from llama_index import StorageContext, load_index_from_storage
from llama_index.prompts import Prompt

from model import service_context

text_qa_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using both the context information and also using your own "
    "knowledge, answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question "
    "on your own.\n"
)
text_qa_template = Prompt(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Using both the new context and your own knowledege, update or "
    "repeat the existing answer.\n"
)
refine_template = Prompt(refine_template_str)

text_qa_template1_str = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Do not give me an answer if it is not mentioned in the context as a fact. \n"
    "Given this information, please provide me with an answer to the following: {query_str}\n"
)
text_qa_template1 = Prompt(text_qa_template1_str)

def run(user_query):
    query = user_query

    storage_context = StorageContext.from_defaults(persist_dir="./data")
    index = load_index_from_storage(
        service_context=service_context, 
        storage_context=storage_context, 
        index_id="db",
        )
    
    engine = index.as_query_engine(
        service_context=service_context,
        text_qa_template=text_qa_template1, 
        refine_template=refine_template,
        streaming=True,
        )
    
    response = engine.query(query)
    print(response)

if __name__ == '__main__':
    query = "what is the dialoGPT mentioned in the DialoGPT paper ?"
    run(query)