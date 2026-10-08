from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.document_service.embeddings import generate_embeddings
from app.services.document_service.vector_store import vector_store


router = APIRouter(
    prefix="/search",
    tags=["Search"]
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("")
async def search_documents(request: SearchRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    if request.top_k < 1:
        raise HTTPException(
            status_code=400,
            detail="top_k must be at least 1"
        )

    # Convert user query into embedding
    query_embedding = generate_embeddings(
        [request.query]
    )[0]
    print("FAISS vector count:", vector_store.count())

    # Search similar document chunks
    results = vector_store.search(
        query_embedding,
        top_k=request.top_k
    )

    return {
        "query": request.query,
        "result_count": len(results),
        "results": results
    }