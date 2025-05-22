from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
        persist_directory: str = "vectorstore_db_faiss"
    ):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
        )
        self.persist_directory = persist_directory

        if os.path.exists(os.path.join(persist_directory, "index.faiss")):
            print("Loading existing FAISS vector store...")
            self.vectorstore = FAISS.load_local(
                persist_directory,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            print("Creating new FAISS vector store...")
            self.vectorstore = None

    def add_documents(self, documents: List[Document], batch_size: int = 4):
        print(f"Starting to add {len(documents)} documents")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i} to {i+len(batch)-1}")
            try:
                for d in batch:
                    print(f"Doc len: {len(d.page_content)}, metadata: {d.metadata}")
                if self.vectorstore is None:
                    # Create new FAISS store from first batch
                    self.vectorstore = FAISS.from_documents(batch, self.embedding_model)
                else:
                    self.vectorstore.add_documents(batch)
                print(f"Successfully added batch {i} to {i+len(batch)-1}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error adding batch: {str(e)}")
                raise

        print("Persisting vector store...")
        self.vectorstore.save_local(self.persist_directory)
        print("Documents added and persisted successfully")

    def search(self, query: str, k: int = 3) -> List[Document]:
        if self.vectorstore is None:
            print("Vector store is empty, no documents to search.")
            return []
        return self.vectorstore.similarity_search(query, k=k)