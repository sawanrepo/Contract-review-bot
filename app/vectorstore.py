import os
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from app.bm25 import BM25Okapi
except ImportError:
    from bm25 import BM25Okapi

_DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_global_embeddings_cache: Dict[str, Any] = {}

def get_shared_embeddings(model_name: str = _DEFAULT_MODEL_NAME):
    if model_name not in _global_embeddings_cache:
        is_bge = "bge" in model_name.lower()
        _global_embeddings_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True if is_bge else False}
        )
    return _global_embeddings_cache[model_name]

class VectorStore:
    """
    Hybrid VectorStore combining FAISS Dense Cosine Retrieval with BM25Okapi.
    Implements F6 configuration:
      - Embedding: BAAI/bge-base-en-v1.5 (768-dim, normalized)
      - Sparse: BM25Okapi with calibrated saturation
      - Fusion: Exact Cosine + BM25 Saturation (alpha=0.6)
      - Default retrieval depth: k=6
    """
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
        self.bm25_index: Optional[BM25Okapi] = None

        model_str = getattr(self.embedding_model, "model_name", "") or embedding_model_name
        self.is_bge = "bge" in str(model_str).lower()

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
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            if "_doc_index" not in doc.metadata:
                doc.metadata["_doc_index"] = start_idx + i
        self.documents.extend(documents)

        # Dense FAISS indexing
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents[:batch_size], self.embedding_model)
            remaining = documents[batch_size:]
        else:
            remaining = documents

        if remaining:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                self.vectorstore.add_documents(batch)

        # Sparse BM25 indexing over complete corpus
        corpus_texts = [d.page_content for d in self.documents]
        self.bm25_index = BM25Okapi(corpus_texts)

        if self.persist_directory:
            try:
                self.vectorstore.save_local(self.persist_directory)
            except Exception as e:
                print(f"Notice: Could not save FAISS to disk: {e}")

    def search(
        self,
        query: str,
        k: int = 6,
        sparse_query: Optional[str] = None,
        alpha: float = 0.6,
        candidate_pool: int = 25,
    ) -> List[Document]:
        scored_results = self.search_with_score(
            query=query,
            k=k,
            sparse_query=sparse_query,
            alpha=alpha,
            candidate_pool=candidate_pool,
        )
        return [doc for doc, _ in scored_results]

    def search_with_score(
        self,
        query: str,
        k: int = 6,
        sparse_query: Optional[str] = None,
        alpha: float = 0.6,
        candidate_pool: int = 25,
        bm25_k: float = 10.0,
    ) -> List[Tuple[Document, float]]:
        if self.vectorstore is None or not self.documents:
            return []

        # Dense query with BGE prompt instruction
        if self.is_bge and not query.startswith("Represent this sentence"):
            dense_q = "Represent this sentence for searching relevant passages: " + query
        else:
            dense_q = query

        # Sparse query (uses clean keyword if provided, otherwise natural query)
        sparse_q = sparse_query if sparse_query else query

        pool = max(k, candidate_pool)

        # 1. Dense retrieval
        dense_hits = self.vectorstore.similarity_search_with_score(dense_q, k=pool)

        # If BM25 is not initialized, return dense results with cosine similarity
        if self.bm25_index is None:
            return [(doc, max(0.0, 1.0 - (float(d) / 2.0))) for doc, d in dense_hits[:k]]

        # 2. Sparse retrieval
        bm25_hits = self.bm25_index.search(sparse_q, k=pool)

        # 3. Hybrid Fusion: Exact Cosine + BM25 Saturation
        # Map documents by content and metadata to deduplicate
        doc_scores: Dict[int, Dict[str, float]] = {}

        # Add dense scores
        # Match doc to index in self.documents
        for doc, l2_dist in dense_hits:
            doc_idx = doc.metadata.get("_doc_index")
            if doc_idx is None or doc_idx >= len(self.documents):
                doc_idx = doc.metadata.get("chunk_index")
            if doc_idx is None or doc_idx >= len(self.documents):
                for idx, d in enumerate(self.documents):
                    if d.page_content == doc.page_content and d.metadata == doc.metadata:
                        doc_idx = idx
                        break
            if doc_idx is None:
                doc_idx = len(self.documents)
                self.documents.append(doc)

            cos_sim = max(0.0, 1.0 - (float(l2_dist) / 2.0))
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = {"dense": 0.0, "sparse": 0.0}
            doc_scores[doc_idx]["dense"] = cos_sim

        # Add sparse scores
        for doc_idx, bm25_raw_score in bm25_hits:
            bm25_sim = max(0.0, float(bm25_raw_score) / (float(bm25_raw_score) + bm25_k))
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = {"dense": 0.0, "sparse": 0.0}
            doc_scores[doc_idx]["sparse"] = bm25_sim

        # Fused combination: alpha * dense + (1 - alpha) * sparse
        fused_ranked: List[Tuple[Document, float]] = []
        for doc_idx, scores in doc_scores.items():
            fused_score = alpha * scores["dense"] + (1.0 - alpha) * scores["sparse"]
            doc = self.documents[doc_idx]
            fused_ranked.append((doc, float(fused_score)))

        fused_ranked.sort(key=lambda x: x[1], reverse=True)
        return fused_ranked[:k]

    def get_all_documents(self) -> List[Document]:
        return list(self.documents)

    def clear(self):
        self.vectorstore = None
        self.bm25_index = None
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
