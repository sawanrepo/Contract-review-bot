from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
        
class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "vectorstore_db"
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)


        self.vectorstore = Chroma(
            collection_name="contract_chunks",
            embedding_function=self.embedding_model,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: List[Document], batch_size: int = 4):
        print(f"Starting to add {len(documents)} documents")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i} to {i+len(batch)-1}")  # More accurate count
            try:
                self.vectorstore.add_documents(batch)
                print(f"Successfully added batch {i} to {i+len(batch)-1}")
            except Exception as e:
                print(f"Error adding batch: {str(e)}")
                raise
        print("Persisting vector store...")
        self.vectorstore.persist()
        print("Documents added successfully")


    def search(self, query: str, k: int = 3) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)