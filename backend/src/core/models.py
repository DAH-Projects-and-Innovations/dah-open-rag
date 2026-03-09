
# ==========================================
# src/core/models.py
# ==========================================

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """Représentation d'un document"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.doc_id is None:
            import uuid
            self.doc_id = str(uuid.uuid4())


@dataclass
class Chunk:
    """Représentation d'un chunk de document"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.chunk_id:
            import uuid
            self.chunk_id = str(uuid.uuid4())


@dataclass
class Query:
    """Représentation d'une requête"""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGResponse:
    """Réponse du pipeline RAG"""
    answer: str
    sources: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)
    query: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la réponse en dictionnaire"""
        return {
            'answer': self.answer,
            'sources': [
                {
                    'content': getattr(s, 'content', str(s)),
                    'metadata': getattr(s, 'metadata', {}) or {},
                    'doc_id': getattr(s, 'doc_id', getattr(s, 'chunk_id', None)),
                    'score': getattr(s, 'score', (getattr(s, 'metadata', {}) or {}).get('score'))
                }
                for s in self.sources
            ],
            'metadata': self.metadata,
            'query': self.query,
            'timestamp': self.timestamp.isoformat()
        }
