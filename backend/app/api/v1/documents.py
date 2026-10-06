from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from app.services.document_service.cleaner import clean_text

from app.services.document_service.extractor import extract_text_from_pdf


router = APIRouter(prefix="/documents", tags=["Documents"])


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and extract its text.
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
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_pdf(str(file_path))
        text = clean_text(raw_text)

        return {
            "filename": file.filename,
            "status": "processed",
            "text_length": len(text),
            "text": clean_text(text)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        await file.close()