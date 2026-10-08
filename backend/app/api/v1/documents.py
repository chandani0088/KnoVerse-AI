from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.services.document_service.extractor import extract_text_from_pdf
from app.services.document_service.cleaner import clean_text
from app.services.document_service.chunker import chunk_text
from app.services.document_service.embeddings import generate_embeddings
from app.services.document_service.vector_store import vector_store


router = APIRouter(prefix="/documents", tags=["Documents"])


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract its text,
    clean the text, split it into chunks,
    generate embeddings, and store them in FAISS.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        # Save uploaded PDF
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text
        raw_text = extract_text_from_pdf(str(file_path))

        # Clean text
        text = clean_text(raw_text)

        # Create chunks
        chunks = chunk_text(text)

        # Generate embeddings for chunks
        embeddings = generate_embeddings(chunks)

        # Store embeddings in FAISS
        vector_store.add_embeddings(
            embeddings,
            chunks
        )

        return {
            "filename": file.filename,
            "status": "processed",
            "text_length": len(text),
            "chunk_count": len(chunks),
            "embedding_dimension": 384,
            "vector_count": vector_store.index.ntotal,
            "chunks": chunks,
            "text": text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        await file.close()