from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
        loader = PyPDFLoader(file_path_to_use,encoding ="utf-8")
    elif extension == '.txt':
        loader = TextLoader(file_path_to_use, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")

    documents = loader.load()
    return documents