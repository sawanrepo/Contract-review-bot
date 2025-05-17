from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

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

    def add_documents(self, documents: List[Document]):
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    def search(self, query: str, k: int = 3) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)