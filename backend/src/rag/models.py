"""
Modèles de données pour le moteur RAG.
Support des réponses structurées avec sources et citations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceLevel(Enum):
    """Niveau de confiance dans la réponse."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class Source:
    """Source d'information citée dans une réponse."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    citation_id: Optional[int] = None  # Numéro de citation [1], [2], etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'relevance_score': self.relevance_score,
            'citation_id': self.citation_id
        }
    
    def get_citation_format(self) -> str:
        """Retourne la source au format de citation."""
        if self.citation_id:
            return f"[{self.citation_id}] {self.content[:200]}..."
        return self.content[:200] + "..."


@dataclass
class Citation:
    """Citation dans une réponse."""
    source_id: str
    citation_number: int
    text_snippet: str
    position_in_answer: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'source_id': self.source_id,
            'citation_number': self.citation_number,
            'text_snippet': self.text_snippet,
            'position_in_answer': self.position_in_answer
        }


@dataclass
class RAGResponse:
    """
    Réponse complète du moteur RAG.
    
    Inclut la réponse générée, les sources utilisées,
    les citations et les métadonnées.
    """
    # Réponse principale
    answer: str
    
    # Sources et citations
    sources: List[Source] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    
    # Métadonnées
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    based_on_context: bool = True  # False si réponse hors contexte
    
    # Informations de génération
    model_used: str = ""
    tokens_used: int = 0
    generation_time_ms: float = 0.0
    
    # Métadonnées additionnelles
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_citations(self) -> bool:
        """Vérifie si la réponse contient des citations."""
        return len(self.citations) > 0
    
    def get_cited_sources(self) -> List[Source]:
        """Retourne uniquement les sources citées."""
        cited_ids = {c.source_id for c in self.citations}
        return [s for s in self.sources if s.id in cited_ids]
    
    def get_formatted_answer(self, include_sources: bool = True) -> str:
        """
        Retourne la réponse formatée avec sources optionnelles.
        
        Args:
            include_sources: Si True, ajoute la liste des sources
            
        Returns:
            Réponse formatée
        """
        formatted = self.answer
        
        if include_sources and self.sources:
            formatted += "\n\nSources:\n"
            for source in self.sources:
                if source.citation_id:
                    formatted += f"\n[{source.citation_id}] {source.content[:200]}..."
                    if 'title' in source.metadata:
                        formatted += f" (Source: {source.metadata['title']})"
        
        return formatted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'answer': self.answer,
            'sources': [s.to_dict() for s in self.sources],
            'citations': [c.to_dict() for c in self.citations],
            'confidence': self.confidence.value,
            'based_on_context': self.based_on_context,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'generation_time_ms': self.generation_time_ms,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """Représentation textuelle."""
        return self.get_formatted_answer(include_sources=True)


@dataclass
class RAGQuery:
    """
    Requête pour le moteur RAG.
    
    Encapsule la question et les paramètres de génération.
    """
    question: str
    
    # Paramètres de retrieval
    top_k: int = 5
    retrieval_filters: Optional[Dict[str, Any]] = None
    
    # Paramètres de génération
    include_citations: bool = True
    max_answer_length: int = 500
    temperature: float = 0.7
    
    # Contexte additionnel
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_context: Optional[str] = None
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'question': self.question,
            'top_k': self.top_k,
            'retrieval_filters': self.retrieval_filters,
            'include_citations': self.include_citations,
            'max_answer_length': self.max_answer_length,
            'temperature': self.temperature,
            'conversation_history': self.conversation_history,
            'user_context': self.user_context,
            'metadata': self.metadata
        }


@dataclass
class RAGConfig:
    """
    Configuration globale du moteur RAG.
    
    Définit le comportement par défaut du système.
    """
    # Mode de fonctionnement
    enable_citations: bool = True
    require_sources: bool = True  # Refuser de répondre sans sources
    
    # Sécurité
    prevent_hallucinations: bool = True
    confidence_threshold: float = 0.5
    
    # Prompts
    system_prompt_template: str = "rag_citations_system"
    user_prompt_template: str = "rag_citations_user"
    
    # Génération
    default_temperature: float = 0.7
    default_max_tokens: int = 500
    
    # Retrieval
    default_top_k: int = 5
    min_relevance_score: float = 0.5
    
    # Formatage
    citation_style: str = "numeric"  # 'numeric' [1], 'author-date', etc.
    include_source_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'enable_citations': self.enable_citations,
            'require_sources': self.require_sources,
            'prevent_hallucinations': self.prevent_hallucinations,
            'confidence_threshold': self.confidence_threshold,
            'system_prompt_template': self.system_prompt_template,
            'user_prompt_template': self.user_prompt_template,
            'default_temperature': self.default_temperature,
            'default_max_tokens': self.default_max_tokens,
            'default_top_k': self.default_top_k,
            'min_relevance_score': self.min_relevance_score,
            'citation_style': self.citation_style,
            'include_source_metadata': self.include_source_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Crée depuis un dictionnaire."""
        return cls(**data)