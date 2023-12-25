from dotenv import load_dotenv

if not load_dotenv():
    print("Cannot load .env file. Environment file is not exists or not readable")
    exit(1)

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from documents_embedding import ingestion
import os
import shutil

def upload_pdf(file_path, destination_dir):
    file_name = file_path.name
    destination_path = destination_dir / file_name

    if not destination_path.exists():
        shutil.copy(file_path, destination_path)
        return f"File '{file_name}' uploaded successfully."
    else:
        return f"File '{file_name}' already exists in the destination directory."

def select_and_upload():
    file_path = Path(filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")]))
    source_directory = Path(os.environ.get('SOURCE_DIRECTORY', 'src_documents'))

    if file_path:
        result_message = upload_pdf(file_path, source_directory)
        status_label.config(text=result_message)

        source_file_path = source_directory / file_path.name
        if source_file_path.exists():
            status_label.config(text=f"File '{file_path.name}' already exists in the source directory.")

        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("PDF Uploader")

    status_label = tk.Label(root, text="")
    status_label.pack()

    select_and_upload()

    root.mainloop()

    ingestion()