# backend/src/api/schemas/request.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class IngestRequest(BaseModel):
    source: str = Field(..., description="Chemin du fichier ou URL")
    loader_name: str = Field(..., description="Nom du loader (ex: pdf_loader)")
    chunker_name: str = Field(..., description="Nom du chunker (ex: overlap_chunker)")
    loader_params: Dict[str, Any] = Field(default_factory=dict)
    chunker_params: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    question: str = Field(..., description="La question de l'utilisateur")
    chat_history: Optional[list] = Field(default_factory=list, description="Historique de la conversation")
    top_k: int = Field(default=5)
    rerank_top_k: Optional[int] = Field(default=None)
    llm_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Paramètres optionnels pour le LLM")
