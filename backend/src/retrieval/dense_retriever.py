"""
Dense Retriever - Récupération basée sur les embeddings vectoriels.
Pass
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from ..core.interfaces import IRetriever, IVectorStore, IEmbedder
from ..core.models import Query, Document


@dataclass
class DenseRetrieverConfig:
    """Configuration pour le Dense Retriever."""
    top_k: int = 10
    similarity_threshold: Optional[float] = None
    include_metadata_filter: bool = True
    normalize_scores: bool = True


class DenseRetriever(IRetriever):
    """
    Dense Retriever utilisant des embeddings vectoriels.
    
    Utilise un vector store pour rechercher les documents les plus similaires
    basé sur la similarité cosine des embeddings.
    """
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedder: IEmbedder,
        config: Optional[DenseRetrieverConfig] = None
    ):
        """
        Initialise le Dense Retriever.
        
        Args:
            vector_store: Store de vecteurs pour la recherche
            embedder: Modèle d'embedding pour encoder les requêtes
            config: Configuration du retriever
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or DenseRetrieverConfig()

    def retrieve(
        self,
        query: Query,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Récupère les documents les plus pertinents via dense retrieval.

        Args:
            query: Requête utilisateur
            top_k: Nombre de documents à récupérer (override config)
            filters: Filtres de métadonnées optionnels

        Returns:
            Liste de documents ordonnés par pertinence
        """
        k = top_k or self.config.top_k

        # Encoder la requête
        query_embedding = self.embedder.embed_query(query.text)

        # Appliquer les filtres de métadonnées si configuré
        search_filters = None
        if self.config.include_metadata_filter and (filters or getattr(query, "filters", None)):
            search_filters = {**(filters or {}), **(query.filters or {})}

        # Recherche vectorielle (le vector store peut retourner des Chunks ou Documents)
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filters=search_filters
        )

        # Normaliser les résultats en objets `Document`
        normalized: List[Document] = []
        for item in raw_results:
            if isinstance(item, Document):
                normalized.append(item)
            else:
                # Supporter Chunk-like objects (duck-typing)
                content = getattr(item, 'content', str(item))
                meta = getattr(item, 'metadata', {}) or {}
                # Si l'objet possède un doc_id ou chunk_id, l'utiliser
                doc_id = getattr(item, 'doc_id', None) or getattr(item, 'chunk_id', None) or meta.get('doc_id')
                doc = Document(content=content, metadata=meta)
                if doc_id:
                    doc.doc_id = doc_id
                # Si un score existe dans metadata, le placer également sur l'objet Document.score
                if 'score' in meta:
                    try:
                        doc.score = float(meta.get('score'))
                    except Exception:
                        doc.score = None
                normalized.append(doc)

        # Filtrer par seuil de similarité si configuré
        if self.config.similarity_threshold is not None:
            normalized = [
                d for d in normalized
                if (getattr(d, 'score', None) or d.metadata.get('score', 0)) >= self.config.similarity_threshold
            ]

        # Normaliser les scores si demandé
        if self.config.normalize_scores and normalized:
            normalized = self._normalize_scores(normalized)

        return normalized

    def _normalize_scores(self, documents: List[Document]) -> List[Document]:
        """
        Normalise les scores entre 0 et 1.

        Args:
            documents: Documents avec scores
            
        Returns:
            Documents avec scores normalisés
        """
        scores = [doc.metadata.get('score', 0) for doc in documents]
        if not scores:
            return documents
        
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            # Tous les scores sont identiques
            for doc in documents:
                doc.metadata['score'] = 1.0
            return documents
        
        # Normalisation min-max
        for doc in documents:
            original_score = doc.metadata.get('score', 0)
            normalized_score = (original_score - min_score) / score_range
            doc.metadata['score'] = normalized_score
            doc.metadata['original_score'] = original_score
        
        return documents
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return {
            'type': 'dense',
            'top_k': self.config.top_k,
            'similarity_threshold': self.config.similarity_threshold,
            'include_metadata_filter': self.config.include_metadata_filter,
            'normalize_scores': self.config.normalize_scores
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Met à jour la configuration dynamiquement.
        
        Args:
            **kwargs: Paramètres de configuration à mettre à jour
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)