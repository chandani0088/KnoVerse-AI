from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.services.document_service.extractor import extract_text_from_pdf
from app.services.document_service.cleaner import clean_text
from app.services.document_service.chunker import chunk_text


router = APIRouter(prefix="/documents", tags=["Documents"])


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract its text,
    clean the text, and split it into chunks.
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

        return {
            "filename": file.filename,
            "status": "processed",
            "text_length": len(text),
            "chunk_count": len(chunks),
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