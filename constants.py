from dotenv import load_dotenv

def load_environment():
    if not load_dotenv():
        print("Cannot load .env file. Environment file is not exists or not readable")
        exit(1)

import os
from chromadb.config import Settings

load_environment()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if PERSIST_DIRECTORY is None:
    raise Exception("Please set the PERSIST_DIRECTORY environment variable")

CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        allow_reset=False,
        anonymized_telemetry=False
)