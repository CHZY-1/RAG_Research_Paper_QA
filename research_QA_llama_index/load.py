import os
from dotenv import load_dotenv, find_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from model import service_context
load_dotenv(find_dotenv())

documents = SimpleDirectoryReader(os.getenv('DOCUMENTS_DIR')).load_data()
index = VectorStoreIndex.from_documents(documents, 
                                        service_context=service_context, 
                                        show_progress=True)
index.set_index_id("db")
index.storage_context.persist("./data")