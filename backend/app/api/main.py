from fastapi import FastAPI

app = FastAPI(
    title="KnoVerse AI",
    description="AI-powered Document Intelligence Platform",
    version="0.1.0",
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