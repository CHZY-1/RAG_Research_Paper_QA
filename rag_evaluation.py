# import torch
# print(torch.cuda.is_available())

import time
import os
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from documents_embedding import get_vector_store, write_data_to_csv
from rag_llm import load_llm

def retrieve_relevant_contexts(vector_db, query_list : list[str], k : int = 3):
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    retrieved_contexts = [retriever.get_relevant_documents(question) for question in query_list]

    questions_context = []

    for question, relevant_documents in zip(query_list, retrieved_contexts):
        page_content = "".join([document.page_content for document in relevant_documents])
        questions_context.append({"question": question, "context": page_content})

    return questions_context

def get_prompt_template(instruction, new_system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_qa_chain(llm):
    qa_sys_prompt = """Answer using the context text provided. Your answers should only answer the question once and not have any text after the answer is done. You answers should avoid starting with 'according to the text' or 'based on the context' If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
    instruction = """CONTEXT:/n/n {context}/n 
    
    Question: {question}""".strip(" ")

    qa_prompt_template = get_prompt_template(instruction, qa_sys_prompt)

    qa_prompt = PromptTemplate(
        template=  qa_prompt_template, 
        input_variables=["context", "question"]
        )
    
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

    return qa_chain

def get_eval_grade_chain(llm):
    PROMPT_TEMPLATE_GRADE = """<s>[INST]You are an expert professor specialized in grading students' answers to questions. You are grading the following question.[/INST]</s>

Question:
{question}
Context:
{context}
[INST]You are grading the following predicted answer:[/INST]
{answer}
[INST]Provide your feedback and grade the predicted answer based on the given context. Use a grading scale from 1 to 5, where 1 is the lowest and 5 is the highest.[/INST]
"""

    prompt_grade = PromptTemplate(input_variables=["question", "context", "answer"], template=PROMPT_TEMPLATE_GRADE)
    eval_grade_chain = LLMChain(llm=llm, prompt=prompt_grade)
    return eval_grade_chain

def get_eval_metrics_chain(llm):
    PROMPT_TEMPLATE_METRICS = """<s>[INST]You are an expert professor specialized in grading students' answers to questions. You are grading the following question.[/INST]</s>

Question:
{question}
Context:
{context}
[INST]You are grading the following predicted answer:[/INST]
{answer}

[INST]Rate the predicted answer based on the given context using the following metrics:

Coherence:
How well does the answer maintain logical and consistent connections with the provided context? Rate from 1 to 5.

Conciseness:
To what extent is the answer clear and to the point within the given context? Rate from 1 to 5.

Accuracy:
Evaluate the accuracy of the answer in relation to the provided context. Rate from 1 to 5.
[/INST]
"""

    prompt_metrics = PromptTemplate(input_variables=["question", "context", "answer"], template=PROMPT_TEMPLATE_METRICS)
    eval_metrics_chain = LLMChain(llm=llm, prompt=prompt_metrics)
    return eval_metrics_chain

if __name__ == "__main__":

    # With context
    config = {'context_length': 4096, 'max_new_tokens': 2048, 'temperature':0.1 ,'repetition_penalty': 1.18, 'gpu_layers':15, 'stream': True}

    llm = load_llm(model='llama2_13b', local= True, config=config)
    critic_llm = load_llm(model='mistral7b_instruct', local= True, config=config)

    dialogpt_query_list = [
        "What is DialoGPT?",
        "What data DialoGPT trained on?", 
        "What are the problems that DialoGPT paper is trying to solve?",
        "How is the architecture of DialoGPT designed to handle conversational context?",
        "Can you explain the key architectural components of DialoGPT mentioned in the paper?",
        "What evaluation metrics were used in the paper to assess the performance of DialoGPT?",
        "Can you discuss the results of the human evaluation mentioned in the paper?",
        "How does DialoGPT compare with other conversation models in terms of performance, according to the paper?",
        "What challenges or limitations does the paper acknowledge in the performance of DialoGPT?",
        "Does the paper discuss any fine-tuning strategies or adaptation techniques for DialoGPT?"
        ]

    transformer_query_list = [
        "What is the fundamental concept introduced in the 'Attention is All You Need' paper?",
        "How does the transformer architecture in the paper handle sequential data, and what advantages does it offer over traditional sequence-to-sequence models?",
        "What specific problems or challenges does the 'Attention is All You Need' paper aim to address in the context of neural machine translation?",
        "Can you provide insights into the self-attention mechanism introduced in the paper and how it contributes to the model's ability to capture long-range dependencies in sequences?",
        "In the 'Attention is All You Need' paper, what are the key components of the transformer model architecture, and how do they work together to process input sequences?",
        "How is positional encoding incorporated into the transformer model, as described in the paper, to account for the sequential nature of input data?",
        "What are the quantitative evaluation metrics used in the paper to assess the performance of the transformer model, and how do they contribute to understanding the model's capabilities?",
        "Can you discuss the empirical results presented in the 'Attention is All You Need' paper, highlighting key findings related to the model's performance on different tasks or datasets?",
        "According to the paper, how does the transformer model compare with earlier models in terms of computational efficiency and parallelization capabilities?",
        "Does the 'Attention is All You Need' paper discuss any potential limitations or challenges associated with the transformer architecture, and what future directions are suggested for further improvement?"
        ]

    vector_db = get_vector_store()

    questions_context = retrieve_relevant_contexts(vector_db, transformer_query_list)

    qa_chain = get_qa_chain(llm=llm)
    eval_grade_chain = get_eval_grade_chain(llm=critic_llm)
    eval_metrics_chain = get_eval_metrics_chain(llm=critic_llm)

    for question_context in questions_context:
        start_time = time.time()

        predictions = qa_chain(question_context)

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        question_context["answer"] = predictions["text"]

        # print(predictions["text"])

        evaluation_grade = eval_grade_chain(question_context)
        evaluation_metrics = eval_metrics_chain(question_context)

        data = {
            "question" : question_context["question"],
            "answer" : predictions["text"],
            "response_time" : response_time,
            "generation_llm" : os.path.basename(llm.model),
            "critic_llm" : os.path.basename(critic_llm.model),
            "evaluation_from_llm_grade" : evaluation_grade['text'].strip(),
            "evaluation_from_llm_metrics": evaluation_metrics['text'].strip()
        }

        write_data_to_csv(data, csv_file_name="rag_llm_assessment.csv")

        
        
        
    # -----------------------------------------------------------------------------------    
    
#     # without context
#     def get_llm_chain_no_context(llm, instruct=True):
#         if instruct:
#             PROMPT_TEMPLATE = """<s>[INST]Answer the following question.[/INST]</s>

# Question:
# {question}
# [/INST]
# """
#         else:
#             PROMPT_TEMPLATE = """Answer the following question.</s>

# Question:
# {question}
# """

#         prompt = PromptTemplate(input_variables=["question"], template=PROMPT_TEMPLATE)
#         llm_chain = LLMChain(llm=llm, prompt=prompt)
#         return llm_chain
    
#     def get_eval_metrics_chain_no_context(llm):
#         PROMPT_TEMPLATE_METRICS = """<s>[INST]You are an expert professor specialized in grading students' answers to questions. You are grading the following question.[/INST]</s>

# Question:
# {question}
# [INST]You are grading the following predicted answer:[/INST]
# {answer}

# [INST]Rate the predicted answer using the following metrics:

# Coherence:
# How well does the answer maintain logical and consistent connections? Rate from 1 to 5.

# Conciseness:
# To what extent is the answer clear and to the point? Rate from 1 to 5.

# Accuracy:
# Evaluate the accuracy of the answer. Rate from 1 to 5.
# [/INST]
# """
#         prompt_metrics = PromptTemplate(input_variables=["question", "answer"], template=PROMPT_TEMPLATE_METRICS)
#         eval_metrics_chain = LLMChain(llm=llm, prompt=prompt_metrics)
#         return eval_metrics_chain



    
#     def evaluate_model_no_context(questions_list, llm, critic_llm):

#         llm_chain_no_context = get_llm_chain_no_context(llm, instruct=False)
#         eval_grade_chain = get_eval_metrics_chain_no_context(llm=critic_llm)

#         for question in questions_list:
#             start_time = time.time()

#             predictions = llm_chain_no_context(question)

#             end_time = time.time()

#             response_time = round(end_time - start_time, 2)

#             eval_dict = {
#                 "question" : question,
#                 "answer" : predictions["text"]
#             }

#             evaluation_grade = eval_grade_chain(eval_dict)

#             data = {
#                 "question" : question,
#                 "answer" : predictions["text"],
#                 "response_time" : response_time,
#                 "generation_llm" : os.path.basename(llm.model),
#                 "critic_llm" : os.path.basename(critic_llm.model),
#                 "evaluation_from_llm_grade" : evaluation_grade['text'].strip(),
#             }

#             write_data_to_csv(data, csv_file_name="rag_llm_assessment_no_context.csv")

#     for questions_list in [dialogpt_query_list, transformer_query_list]:
#         evaluate_model_no_context(questions_list, llm=llm, critic_llm=critic_llm)



    

