import time, os
from documents_embedding import get_vector_store, write_data_to_csv
from rag_llm import load_llm
from rag_evaluation import retrieve_relevant_contexts, get_qa_chain, get_eval_grade_chain, get_eval_metrics_chain, get_prompt_template
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

def combine_dicts(list1, list2, key='question'):
    # combined_dict = {item[key]: {**item, **next((d for d in list2 if d[key] == item[key]), {})} for item in list1}

    combined_dict = [
        {
            'question': item['question'],
            'ground-truth': item['answer'],
            'context': next((d['context'] for d in list2 if d[key] == item[key]), None)
        }
        for item in list1
    ]
    return combined_dict

def get_qa_chain(llm):
    qa_sys_prompt = """Answer using the context text provided. Your answers should only answer the question once and not have any text after the answer is done."""
    instruction = """CONTEXT:/n/n {context}/n 
    
    Question: {question}""".strip(" ")

    qa_prompt_template = get_prompt_template(instruction, qa_sys_prompt)

    qa_prompt = PromptTemplate(
        template=  qa_prompt_template, 
        input_variables=["context", "question"]
        )
    
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

    return qa_chain

if __name__ == "__main__":

    open_domain_questions = [
        {
            'question': "What is the capital of Australia?",
            'answer': "The capital of Australia is Canberra."
        },
        {
            'question': "How does photosynthesis work in plants?",
            'answer': "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight into energy to produce glucose and oxygen."
        },
        {
            'question': "Who is the author of 'To Kill a Mockingbird'?",
            'answer': "Harper Lee is the author of 'To Kill a Mockingbird.'"
        },
        {
            'question': "What is the main function of the human respiratory system?",
            'answer': "The main function of the human respiratory system is to facilitate the exchange of oxygen and carbon dioxide between the body and the environment."
        },
        {
            'question': "What is the purpose of the International Space Station (ISS)?",
            'answer': "The International Space Station (ISS) serves as a microgravity and space environment research laboratory, where scientific research is conducted in astrobiology, astronomy, and other fields."
        },
        {
            'question': "Who painted the famous artwork 'Starry Night'?",
            'answer': "Vincent van Gogh painted the famous artwork 'Starry Night.'"
        },
        {
            'question': "What are the three states of matter?",
            'answer': "The three states of matter are solid, liquid, and gas."
        },
        {
            'question': "What is the significance of the Turing Test in artificial intelligence?",
            'answer': "The Turing Test is a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human. It is often used to assess the capability of a machine to engage in natural language conversation."
        },
        {
            'question': "How does a typical computer hard drive store data?",
            'answer': "A typical computer hard drive stores data using magnetization, where binary data is represented by the orientation of magnetic domains on the disk's surface."
        },
        {
            'question': "Who developed the theory of relativity?",
            'answer': "Albert Einstein developed the theory of relativity, with the famous equations E=mc² representing the equivalence of energy (E) and mass (m)."
        }
    ]

    config = {'context_length': 4096, 'max_new_tokens': 2048, 'temperature':0.1 ,'repetition_penalty': 1.18, 'gpu_layers':20, 'stream': True}

    llm = load_llm(model='llama2_7b_base', local= True, config=config)
    critic_llm = load_llm(model='mistral7b_instruct', local= True, config=config)

    vector_db = get_vector_store()

    questions = [qa['question'] for qa in open_domain_questions]

    questions_context = retrieve_relevant_contexts(vector_db, questions)

    questions_answer_context = combine_dicts(open_domain_questions, questions_context, key="question")

    qa_chain = get_qa_chain(llm=llm)
    eval_grade_chain = get_eval_grade_chain(llm=critic_llm)
    eval_metrics_chain = get_eval_metrics_chain(llm=critic_llm)

    for i, question_context in enumerate(questions_answer_context):
        start_time = time.time()

        predictions = qa_chain(question_context)

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        question_context["answer"] = predictions["text"]

        evaluation_grade = eval_grade_chain(question_context)
        evaluation_metrics = eval_metrics_chain(question_context)

        data = {
            "question" : question_context["question"],
            "answer" : predictions["text"],
            "ground-truth" : question_context["ground-truth"],
            "response_time" : response_time,
            "generation_llm" : os.path.basename(llm.model),
            "critic_llm" : os.path.basename(critic_llm.model),
            "evaluation_from_llm_grade" : evaluation_grade['text'].strip(),
            "evaluation_from_llm_metrics": evaluation_metrics['text'].strip(),
            "context-retreived" : question_context["context"],
        }

        write_data_to_csv(data, csv_file_name="rag_llm_assessment_open_domain.csv")