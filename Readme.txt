Install all the libraries to run this

. Recommend to create a virtual environment.
. Install all the libraries
. Activate virtual env
. Use command "panel serve ui.py" to start the server on localhost
. Access the local host url appears in the console
. For example, 'Bokeh app running at: http://localhost:5006/ui'.


# Creating Knowledge Base

- Two Options:

  1. Manual:
     - Copy and Paste all research papers in the src_documents folder.
     - Execute the 'ingestion()' function in documents_embeddings.py to process, chunk, and embed the documents. This script ensures the data is ready for the knowledge base.

  2. User Interface Upload:
     - Run the ui_upload_pdf.py script.
     - Use the user interface to upload research paper into the src_documents folder.
     - Execute the ingestion() function in documents_embeddings.py to process, chunk, and embed the uploaded documents.




# Running the RAG Application:

1. Using Command Panel to Serve UI:
   - Start the UI by running the command 'panel serve ui.py'.
   - This launches the RAG application with a user interface for interactive use.

2. Manually Running rag.py with Query:
   - Alternatively, run 'rag.py' manually.
   - Pass in the query at the end of the python file to execute the RAG application without the user interface.


Python Libraries:
langchain -> RAG
ctransformers[cuda] -> support GPU
ctransformers -> CPU only
sentence-transformers -> embeddings
chromadb -> vector database
python-dotenv -> load variable from .env file
panel -> UI

command: pip install langchain ctransformers[cuda] sentence-transformers chromadb panel python-dotenv


LLMs Usage Guide:

1. Default Model:
   - The default model is Llama2-chat 7b with 4bits quantization.
   - To use another model, refer to the available model types in 'rag_llm.py'.
   - The keys in MODEL_KWARGS, such as 'llama2_7b', can be passed as a parameter to the load_llm() function to specify the desired LLM.

2. Downloading LLMs from Hugging Face:
   - There are two options to download LLMs from Hugging Face instead of loading them locally:
     a. In 'rag_llm.py', set the 'local' key in MODEL_KWARGS dictionary to False.
     b. Explicitly pass the 'local' parameter as False into the load_llm() function while keeping the MODEL_KWARGS 'local' key unchanged.

3. Loading LLMs from Local Directory:
   - To load LLMs stored in a local directory, modify the 'MODEL_KWARGS' 'model_file' key with the specific LLM path and set 'local' to True in 'rag_llm.py'.



## GPU Configuration in rag_llm.py

If you do not have a GPU available, configure the gpu_layers variable in the rag_llm.py file appropriately. Follow these steps:

1. Open the rag_llm.py file.
2. Locate the CONFIG variable in the script.
3. Find the gpu_layers parameter in the CONFIG dictionary.
4. Set the gpu_layers value to 0.



LLM Configuration Instructions:

1. Changing Default Model (Llama2-chat 7b):
   - To change the default model (Llama2-chat 7b) in the application, access 'ui.py'.
   - Specify the desired model in the 'get_rag_chain_async()' function parameter within 'ui.py'.

2. Location of RAG Functions:
   - All RAG-related functions are located in the 'rag.py' file.


example:

def _get_llm_chain():
    llm_chain = get_rag_chain_async(model='mistral7b_instruct', local=False)
    return llm_chain



Directory

- RAG_Research_paper: Main directory for the RAG research paper application.
	- .env: File storing all environment variables necessary for running the RAG application.
	- constant.py: File storing configurations for the vector database, essential for the proper functioning of the RAG system.
	- db (vector db): Directory containing the vector database used during the execution of the RAG application.
	- src_documents: Directory containing all research papers in PDF format, used as source documents for queries in the RAG system.
	- summaries
	- documents_embeddings.py: Script for data preprocessing such as loading pdf, chunking, and embeddings and store in vector database
	- rag_evaluation.py: Script for conducting RAG evaluations.
	- rag_llm.py: Script handling the loading and configuration of LLMs for the RAG system.
	- ui_upload.py: Script running a tkinter user interface to upload research paper PDFs into the src_documents directory.
	- ui.py: Script running the main user interface for the RAG application.
	- rag_research_paper.csv: CSV file used to record information when the user executes the rag_chain() function in rag.py.
	- rag_llm_assessment.csv: CSV file for storing RAG evaluation results.

