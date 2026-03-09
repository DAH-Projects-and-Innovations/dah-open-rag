"""
Module RAG - Moteur de génération avec citations et contrôle.
"""

from .models import (
    RAGResponse,
    RAGQuery,
    RAGConfig,
    Source,
    Citation,
    ConfidenceLevel
)

from .engine import (
    RAGEngine,
    SimpleRAG,
    CitationRAG
)

__all__ = [
    # Models
    'RAGResponse',
    'RAGQuery',
    'RAGConfig',
    'Source',
    'Citation',
    'ConfidenceLevel',
    
    # Engines
    'RAGEngine',
    'SimpleRAG',
    'CitationRAG'
]