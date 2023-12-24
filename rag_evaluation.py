# import torch
# print(torch.cuda.is_available())

import os
from dotenv import load_dotenv
from documents_embedding import load_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from rag import load_llm, get_vector_store, write_data_to_csv

if not load_dotenv():
    print("Cannot load .env file. Environment file is not exists or not readable")
    exit(1)

persist_directory = os.environ.get('PERSIST_DIRECTORY')
# source_directory = os.environ.get('SOURCE_DIRECTORY', 'src_documents')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'evaluation')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

models_path = '../models/'

config = {'context_length': 4096, 'max_new_tokens': 2048, 'temperature':0.1 ,'repetition_penalty': 1.1, 'gpu_layers':20}
# llm = CTransformers(model=models_path+'yarn-mistral-7b-128k.Q4_K_M.gguf', model_type='mistral', config=config)

# print(llm.config['temperature'])
critic_llm = CTransformers(model=models_path+'mistral-7b-instruct-v0.1.Q4_K_M.gguf', model_type='mistral', config=config)
llm = load_llm(model='llama')

documents = load_documents(source_dir=source_directory)

embedding_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)

query_list = [
        "What data DialoGPT trained on?", 
        "What is the problems that DialoGPT paper trying to solve?", 
        "What is DialoGPT?",
        # "How is the architecture of DialoGPT designed to handle conversational context?",
        # "Can you explain the key architectural components of DialoGPT mentioned in the paper?",
        # "What evaluation metrics were used in the paper to assess the performance of DialoGPT?",
        # "Can you discuss the results of the human evaluation mentioned in the paper?",
        # "How does DialoGPT compare with other conversation models in terms of performance, according to the paper?",
        # "What challenges or limitations does the paper acknowledge in the performance of DialoGPT?",
        # "Does the paper discuss any fine-tuning strategies or adaptation techniques for DialoGPT?"
        ]

vector_db = get_vector_store()

retriever = vector_db.as_retriever(search_kwargs={"k":3})

retrieved_contexts = [retriever.get_relevant_documents(question) for question in query_list]

questions_context = []

for question in query_list:
    # retreive documents list for every question
    relevant_documents = retriever.get_relevant_documents(question)
    page_content = ""
    # loop through document object
    for document in relevant_documents:
        page_content += document.page_content
        
    questions_context.append({"question": question, "context": page_content})

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import time

def get_prompt_template(instruction, new_system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_qa_prompt():
    sys_prompt = """Answer using the context text provided. Your answers should only answer the question once and not have any text after the answer is done. You answers should avoid starting with 'according to the text' or 'based on the context' If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
    instruction = """CONTEXT:/n/n {context}/n 
    
    Question: {question}""".strip(" ")

    prompt_template = get_prompt_template(instruction, sys_prompt)

    prompt = PromptTemplate(
        template= prompt_template, 
        input_variables=["context", "question"]
        )

    return prompt


# QA_PROMPT = "Answer the question based on the context\nContext:{context}\nQuestion:{question}\nAnswer:"

template  = get_qa_prompt()
# template = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT)
qa_chain = LLMChain(llm=llm, prompt=template)
# predictions = qa_chain.apply(questions_context)


PROMPT_TEMPLATE = """<s>[INST]You are an expert professor specialized in grading students' answers to questions. You are grading the following question.[/INST]</s>

Question:
{question}
Context:
{context}
[INST]You are grading the following predicted answer:[/INST]
{answer}
[INST]Provide your feedback and grade the predicted answer based on the given context. Use a grading scale from 1 to 5, where 1 is the lowest and 5 is the highest.[/INST]
"""
 
PROMPT = PromptTemplate(input_variables=["question", "context", "answer"], template=PROMPT_TEMPLATE)

eval_chain_llm = LLMChain(llm=critic_llm, prompt=PROMPT)

from langchain.evaluation.qa import ContextQAEvalChain
from langchain.evaluation.criteria import CriteriaEvalChain, LabeledCriteriaEvalChain
# eval_chain = ContextQAEvalChain.from_llm(llm, verbose=True)

criteria = {
        "hallucination": (
            "Does this submission contain information"
            " not present in the input or reference?"
        ),
    }

eval_chain =CriteriaEvalChain.from_llm(llm=critic_llm, criteria="coherence")
# graded_outputs = eval_chain.evaluate(questions_context, predictions, question_key="question", prediction_key="text")


for question_context in questions_context:
    start_time = time.time()

    predictions = qa_chain(question_context)
    # print(predictions)

    response_time = time.time() - start_time

    question_context["answer"] = predictions["text"]
    question_context["response_time"] = response_time

    evaluation = eval_chain_llm(question_context)
    question_context["evaluation_from_llm"] = evaluation['text']
    print(question_context["evaluation_from_llm"])
    question_context["critic_llm"] = os.path.basename(critic_llm.model)

    # result = eval_chain.evaluate_strings(prediction=predictions["text"], input=question_context["question"])
    # write_data_to_csv(question_context, csv_file_name="eval_rag_research_paper.csv")
    
    print(question_context)
