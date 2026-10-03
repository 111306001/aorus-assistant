# src/vector_store.py
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # 使用輕量級 Embedding 模型（會在 CPU 運行，不佔用 GPU VRAM）
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, docs: list[str]):
        self.documents = docs
        print("Encoding documents into vectors...")
        self.embeddings = self.model.encode(docs, show_progress_bar=False)

    def search(self, query: str, top_k: int = 2) -> list[str]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        query_embedding = self.model.encode([query])[0]
        # 計算 Cosine Similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]