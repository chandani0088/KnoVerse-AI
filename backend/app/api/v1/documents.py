from typing import Annotated
from pathlib import Path
from uuid import uuid4
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_service.extractor import extract_text_from_pdf
from app.services.document_service.cleaner import clean_text
from app.services.document_service.chunker import chunk_text
from app.services.document_service.embeddings import generate_embeddings
from app.services.document_service.vector_store import vector_store


router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def process_pdf(file: UploadFile) -> dict:
    """Save, process, and index one PDF document."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    original_filename = Path(file.filename).name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported: {original_filename}",
        )

    # Unique disk filename prevents overwriting another upload.
    saved_filename = f"{uuid4().hex}.pdf"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_pdf(str(file_path))
        text = clean_text(raw_text)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No extractable text found in {original_filename}. "
                       "Scanned PDFs may need OCR.",
            )

        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No text chunks generated for {original_filename}",
            )

        embeddings = generate_embeddings(chunks)

        vector_store.add_embeddings(
            embeddings,
            chunks,
            metadatas=[
                {"filename": original_filename}
                for _ in chunks
            ],
        )

        return {
            "filename": original_filename,
            "status": "processed",
            "text_length": len(text),
            "chunk_count": len(chunks),
            "embedding_dimension": 384,
            "total_vectors_in_store": vector_store.count(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed for {original_filename}: {exc}",
        ) from exc
    finally:
        await file.close()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process one PDF."""
    return await process_pdf(file)


@router.post("/upload-multiple")
async def upload_multiple_documents(
    files: Annotated[list[UploadFile], File(
        description="Select one or more PDF files"
    )],
):
    """Upload and process multiple PDFs."""

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one PDF",
        )

    results = []

    for file in files:
        try:
            result = await process_pdf(file)
            results.append(result)
        except HTTPException as exc:
            results.append({
                "filename": file.filename or "unknown",
                "status": "failed",
                "error": exc.detail,
            })

    successful = sum(
        1 for result in results if result["status"] == "processed"
    )

    return {
        "total_files": len(files),
        "successful_files": successful,
        "failed_files": len(files) - successful,
        "total_vectors_in_store": vector_store.count(),
        "results": results,
    }
