
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np


class VectorStore:
    """Persistent FAISS vector store for semantic search."""

    def __init__(
        self,
        dimension: int = 384,
        storage_dir: str = "storage/vector_store",
    ):
        self.dimension = dimension
        self.storage_dir = Path(storage_dir).resolve()

        self.index_path = self.storage_dir / "index.faiss"
        self.metadata_path = self.storage_dir / "documents.json"

        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[str] = []
        self.metadatas: List[dict] = []

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict] | None = None,
    ):
        """Store embeddings, text chunks, and metadata."""

        if len(embeddings) != len(texts):
            raise ValueError(
                "Number of embeddings must match number of texts"
            )

        if not embeddings:
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError(
                "Number of metadata entries must match number of texts"
            )

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Invalid embedding dimensions")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values")

        self.index.add(vectors)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        self.save()

    def save(self):
        """Persist the FAISS index and document metadata."""

        if self.index.ntotal != len(self.documents):
            raise RuntimeError("FAISS index and documents are out of sync")

        if len(self.documents) != len(self.metadatas):
            raise RuntimeError("Documents and metadata are out of sync")

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        payload = {
            "dimension": self.dimension,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self):
        """Load previously saved vectors and metadata."""

        if not self.index_path.exists() and not self.metadata_path.exists():
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise RuntimeError(
                "Vector store files are incomplete. "
                "Back up the storage folder before repairing it."
            )

        saved_index = faiss.read_index(str(self.index_path))

        payload = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

        if payload.get("dimension") != self.dimension:
            raise ValueError("Saved vector dimension does not match")

        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])

        if saved_index.d != self.dimension:
            raise ValueError("Saved FAISS index has an invalid dimension")

        if saved_index.ntotal != len(documents):
            raise ValueError("Saved index and documents are out of sync")

        if len(documents) != len(metadatas):
            raise ValueError("Saved documents and metadata are out of sync")

        self.index = saved_index
        self.documents = documents
        self.metadatas = metadatas
        
    def has_document_hash(self, file_hash: str) -> bool:
        """Check whether a document with this file hash is already indexed."""
        return any(
            metadata.get("file_hash") == file_hash
            for metadata in self.metadatas
        )
    
    def count(self) -> int:
        """Return the number of stored vectors."""
        return self.index.ntotal

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ):
        """Search for the most similar text chunks."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if (
            query_vector.shape != (1, self.dimension)
            or not np.isfinite(query_vector).all()
        ):
            raise ValueError("Invalid query embedding")

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append(
                {
                    "text": self.documents[idx],
                    "score": float(score),
                    "metadata": self.metadatas[idx],
                }
            )

        return results


# Shared persistent vector store instance
vector_store = VectorStore()
