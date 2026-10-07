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

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str]
    ):
        """
        Add embeddings and their corresponding text chunks.
        """

        if not embeddings:
            return

        if len(embeddings) != len(texts):
            raise ValueError(
                "Number of embeddings must match number of texts"
            )

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)
        self.documents.extend(texts)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ):
        """
        Search for the most similar text chunks.
        """

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                "text": self.documents[index],
                "score": float(score)
            })

        return results