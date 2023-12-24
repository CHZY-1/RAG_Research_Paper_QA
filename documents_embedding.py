from dotenv import load_dotenv

if not load_dotenv():
    print("Cannot load .env file. Environment file is not exists or not readable")
    exit(1)


import os
import glob
import pandas as pd
from datetime import datetime as dt
from pathlib import Path as p
from tqdm import tqdm
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from rag_llm import load_llm
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
        
    filtered_files = get_files_in_dir(source_dir, ignored_files)

    results = []
    with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
        for file_path in filtered_files:
            docs = load_single_document(file_path)
            if docs:
                results.extend(docs)
                pbar.update()

    return results

def get_files_in_dir(source_dir: str, ignored_files: List[str] = []):

    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
            )
        
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    return filtered_files


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

def process_single_document(file_path, chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    document = load_single_document(file_path)

    if not document:
        print("No new document to load")
        return None
    
    text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    return texts


def process_documents(source_dir : str = source_directory, chunk_size: int = 500, chunk_overlap: int = 20, ignored_files: List[str] = []) -> List[Document]:
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

def log_summaries(map_reduce_outputs, source, output_dir = "summaries"):
    final_mp_data = []
    for doc, out in zip(
        map_reduce_outputs["input_documents"], map_reduce_outputs["intermediate_steps"]
    ):
        output = {}
        output["file_name"] = p(doc.metadata["source"]).stem
        output["file_type"] = p(doc.metadata["source"]).suffix
        output["page_number"] = doc.metadata["page"]
        output["chunks"] = doc.page_content
        output["concise_summary"] = out
        final_mp_data.append(output)

    final_mp_data.append({
        "file_name": "",
        "file_type": "",
        "page_number": "",
        "chunks": "",
        "concise_summary": map_reduce_outputs['output_text']
    })

    pdf_mp_summary = pd.DataFrame.from_dict(final_mp_data)
    pdf_mp_summary = pdf_mp_summary.sort_values(
        by=["file_name", "page_number"]
    )  # sorting the dataframe by filename and page_number
    pdf_mp_summary.reset_index(inplace=True, drop=True)

    timestamp = dt.now().strftime(r"%Y%m%d%H%M%S")
    output_file = p(output_dir) / f"{timestamp}_{p(source).stem}_summary.csv"
    pdf_mp_summary.to_csv(output_file, index=False)


def create_summary(file_path, chunk_size = 800, chunk_overlap= 50):

    texts = process_single_document(file_path=file_path , chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    summarize_chain = load_summarize_chain(
        llm = load_llm(),
        chain_type='map_reduce',
        return_intermediate_steps=True,
        verbose=False
    )
    source = os.path.basename(file_path)

    print(f"Creating summary for {source} ...")
    summaries = summarize_chain(texts)
    log_summaries(summaries, source)

    return summaries['output_text'], source

def ingest_summary(file_path):

    text, source = create_summary(file_path)

    new_doc =  [Document(page_content=text, metadata={"source": source, "summarize": "Yes"})]

    embedding_model = get_embedding_model()

    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)

    
    if text is not None:

        if vectorstore_exist(persist_directory, embedding_model, CHROMA_SETTINGS):

            print(f"Appending to existing vectorstore at {persist_directory}")
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model, client_settings=CHROMA_SETTINGS, client=chroma_client)
            db.add_documents(new_doc)

        else:
            print("Creating new vectorstore")
            db = Chroma.from_documents(new_doc, embedding_model, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)

    if text:
        db.persist()
        db = None

        print(f"Ingestion complete for {source}.")
    else:
        print("Nothing to ingest.")


def ingest_summaries_from_dir(source_dir : str):

    all_files = get_files_in_dir(source_dir)

    for file_path in all_files:
        ingest_summary(file_path)

    print(f"Ingestions for all files in {source_dir} have completed.")


def get_embedding_model(embedding_model_name=embeddings_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return embedding_model


def ingestion():

    embeddings = get_embedding_model()
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
            # filter out all the files already existing in the vector db.
            ignored_files=[metadata['source'] for metadata in collection['metadatas'] 
                           if metadata and ('summarize' not in metadata or metadata['summarize'].lower() != 'yes')]
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

    # chunk all the files in a directory and store in vector db
    # ingestion()

    # summarize all the files in a directory and store in vector db
    ingest_summaries_from_dir(source_dir=source_directory)

    # create_summary("src_documents\DIALOGPT.pdf")

