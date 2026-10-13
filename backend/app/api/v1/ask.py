
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service.rag_service import rag_service

router = APIRouter(prefix="/ask", tags=["Ask AI"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("")
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        return rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check backend logs."
        )
