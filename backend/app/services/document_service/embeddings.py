from sentence_transformers import SentenceTransformer
from typing import List


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert text chunks into numerical embeddings.

    Args:
        texts: List of text chunks.

    Returns:
        List of embedding vectors.
    """

    if not texts:
        return []

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()