from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents(file_path):
    if hasattr(file_path, 'name') and hasattr(file_path, 'getbuffer'):
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file_path.name)
        with open(temp_file_path, "wb") as f:
            f.write(file_path.getbuffer())
        file_path_to_use = temp_file_path
    elif isinstance(file_path, str):
        file_path_to_use = file_path
    else:
        raise ValueError("Invalid input. Must be a file path string or uploaded file.")

    extension = os.path.splitext(file_path_to_use)[1].lower()
    if extension == '.pdf':
        loader = PyPDFLoader(file_path_to_use)
        documents = loader.load()
        return documents
    elif extension == '.txt':
        loader = TextLoader(file_path_to_use, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")