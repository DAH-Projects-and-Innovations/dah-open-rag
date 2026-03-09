# backend/src/api/routes/ingest.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pathlib import Path
import shutil
import uuid
from typing import List

from src.Chunkers.basic_chunker import ConfigurableChunker
from src.Loaders.text_loader import UnifiedDocumentLoader
from src.api.dependencies import get_pipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

SUPPORTED_EXTENSIONS = {f".{fmt}" for fmt in UnifiedDocumentLoader().get_supported_formats()}

BASE_UPLOAD_DIR = Path("data/uploads")
INSTANCE_UPLOAD_DIR = BASE_UPLOAD_DIR / str(uuid.uuid4())
INSTANCE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("")
async def ingest_uploaded_files(
    files: List[UploadFile] = File(..., description="Upload un ou plusieurs fichiers (PDF, TXT, MD)"),
    pipeline=Depends(get_pipeline),
):
    """
    Upload de fichiers, sauvegarde, puis ingestion dans le vector store.
    Formats supportés : PDF, TXT, MD.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé.")

    upload_dir = INSTANCE_UPLOAD_DIR
    saved_files: List[str] = []

    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        safe_name = f"{uuid.uuid4()}{suffix}"
        file_path = upload_dir / safe_name

        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_files.append(f.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde de {f.filename}: {str(e)}")
        finally:
            await f.close()

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail=f"Aucun fichier valide. Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        chunks_count = pipeline.ingest(
            loader=UnifiedDocumentLoader(),
            chunker=ConfigurableChunker(),
            source=str(upload_dir)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ingestion : {str(e)}")

    return {
        "status": "success",
        "documents_processed": len(saved_files),
        "chunks_ingested_total": int(chunks_count),
        "files": saved_files,
    }
