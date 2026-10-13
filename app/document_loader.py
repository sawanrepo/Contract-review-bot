from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os

def load_documents(file_path) -> List[Document]:
    is_temp_created = False
    if hasattr(file_path, 'name') and hasattr(file_path, 'getbuffer'):
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.basename(file_path.name)
        temp_file_path = os.path.join(temp_dir, f"upload_{os.getpid()}_{base_name}")
        with open(temp_file_path, "wb") as f:
            f.write(file_path.getbuffer())
        file_path_to_use = temp_file_path
        is_temp_created = True
    elif isinstance(file_path, str):
        file_path_to_use = file_path
    else:
        raise ValueError("Invalid input. Must be a file path string or uploaded file.")

    try:
        extension = os.path.splitext(file_path_to_use)[1].lower()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=900,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False
        )

        if extension == '.pdf':
            loader = PyPDFLoader(file_path_to_use)
            raw_documents = loader.load()
            # Normalize page metadata to 1-based 'page_number'
            for doc in raw_documents:
                page_idx = doc.metadata.get("page", 0)
                doc.metadata["page_number"] = page_idx + 1
            return text_splitter.split_documents(raw_documents)
        elif extension == '.txt':
            loader = TextLoader(file_path_to_use, encoding="utf-8")
            raw_documents = loader.load()
            for doc in raw_documents:
                doc.metadata["page_number"] = 1
            return text_splitter.split_documents(raw_documents)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
    finally:
        if is_temp_created and os.path.exists(file_path_to_use):
            try:
                os.remove(file_path_to_use)
            except Exception as e:
                print(f"Notice: Failed to clean up temp file {file_path_to_use}: {e}")