from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_global_embeddings_cache = {}

def get_shared_embeddings(model_name: str = _DEFAULT_MODEL_NAME):
    if model_name not in _global_embeddings_cache:
        _global_embeddings_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
    return _global_embeddings_cache[model_name]

class VectorStore:
    def __init__(
        self,
        embedding_model=None,
        embedding_model_name: str = _DEFAULT_MODEL_NAME,
        persist_directory: Optional[str] = None
    ):
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = get_shared_embeddings(embedding_model_name)

        self.persist_directory = persist_directory
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []

        if persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(
                    persist_directory,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                print(f"Notice: Could not load FAISS from {persist_directory}: {e}")
                self.vectorstore = None

    def add_documents(self, documents: List[Document], batch_size: int = 16):
        if not documents:
            return
        self.documents.extend(documents)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents[:batch_size], self.embedding_model)
            remaining = documents[batch_size:]
        else:
            remaining = documents

        if remaining:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                self.vectorstore.add_documents(batch)

        if self.persist_directory:
            try:
                self.vectorstore.save_local(self.persist_directory)
            except Exception as e:
                print(f"Notice: Could not save FAISS to disk: {e}")

    def search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        return list(self.documents)

    def clear(self):
        self.vectorstore = None
        self.documents = []
        if self.persist_directory:
            index_path = os.path.join(self.persist_directory, "index.faiss")
            store_path = os.path.join(self.persist_directory, "index.pkl")
            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(store_path):
                    os.remove(store_path)
            except Exception as e:
                print(f"Notice during clear: {e}")