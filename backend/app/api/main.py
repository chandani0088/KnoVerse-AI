from fastapi import FastAPI

from app.api.v1.documents import router as documents_router
from app.api.v1.search import router as search_router


app = FastAPI(
    title="KnoVerse AI",
    description="AI-powered Document Intelligence Platform",
    version="0.1.0",
)


app.include_router(
    documents_router,
    prefix="/api/v1"
)

app.include_router(
    search_router,
    prefix="/api/v1"
)


@app.get("/")
def root():
    return {
        "message": "KnoVerse AI API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }