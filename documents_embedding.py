from dotenv import load_dotenv

if not load_dotenv():
    print("Cannot load .env file. Environment file is not exists or not readable")
    exit(1)


import os
import glob
from tqdm import tqdm
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from constants import CHROMA_SETTINGS
import chromadb

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    PyPDFLoader
)


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'src_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]: 
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
            )
        
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    results = []
    with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
        for file_path in filtered_files:
            docs = load_single_document(file_path)
            if docs:
                results.extend(docs)
                pbar.update()

    return results

def get_text_splitter(chunk_size : int = 500, chunk_overlap : int =0, separators : list[str] = ["\n"], type : str = 'recursive'):
    
    if type == "char":
        text_splitter = CharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len)
    
    return text_splitter


def process_documents(chunk_size: int = 500, chunk_overlap: int = 20, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    #filter empty string in list
    documents = list(filter(None, documents))
    if not documents:
        print("No new documents to load")
        return None
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # print(documents)
    text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    
    return texts

def vectorstore_exist(persist_directory: str, embeddings: HuggingFaceEmbeddings, settings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=settings)
    if not db.get()['documents']:
        return False
    return True

def ingestion():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Chroma persistent client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)

    if vectorstore_exist(persist_directory, embeddings, CHROMA_SETTINGS):
        # update vector db
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        # print(collection)
        texts = process_documents(
            chunk_size=800,
            chunk_overlap=50,
            ignored_files=[metadata['source'] for metadata in collection['metadatas']]
            )
        if texts is not None:
            print("Creating embeddings...")
            db.add_documents(texts)
    else:
        # Create and store in vector db
        print("Creating new vectorstore")
        texts = process_documents(
            chunk_size=800,
            chunk_overlap=50
            )
        print("Creating embeddings...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)

    if texts:
        db.persist()
        db = None

        print("Ingestion complete.")
    else:
        print("Nothing to ingest.")


if __name__ == "__main__":
    ingestion()

