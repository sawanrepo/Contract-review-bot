from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_documents(file_path):
    if hasattr(file_path, 'name'):
        filename = file_path.name
    elif isinstance(file_path, str): #added to handle string input if file_path is a string (just to check fxn working in terminal)
        filename = file_path
    else:
        filename = ''

    extension = os.path.splitext(filename)[1].lower()
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")

    documents = loader.load()
    return documents


# just checking .....
if __name__ == "__main__":  
    
    file_path = "D:\\Contract-review-bot\\app\\2024BCS0164 SAWAN KUMAR (1).pdf" 
    documents = load_documents(file_path)
    print(documents[0])