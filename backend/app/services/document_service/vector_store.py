import faiss
import numpy as np
from typing import List


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

        # Inner Product index
        # Works with normalized embeddings for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Keep original text chunks
        self.documents: List[str] = []
        self.metadatas: List[dict] = []

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict] | None = None
    ):
        """Store embeddings, text chunks, and metadata."""

        if not embeddings:
            return

        if len(embeddings) != len(texts):
            raise ValueError(
                "Number of embeddings must match number of texts"
            )

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError(
                "Number of metadata entries must match number of texts"
            )

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Invalid embedding dimensions")

        self.index.add(vectors)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def count(self) -> int:
        """
        Return the number of stored vectors.
        """
        return self.index.ntotal

    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ):
        """Search for the most similar text chunks."""

        if self.index.ntotal == 0:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Invalid query embedding dimensions")

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append({
                "text": self.documents[idx],
                "score": float(score),
                "metadata": self.metadatas[idx]
            })

        return results

# Shared vector store instance
vector_store = VectorStore()