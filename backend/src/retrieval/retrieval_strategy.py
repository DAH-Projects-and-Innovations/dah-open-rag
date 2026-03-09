"""
Système de stratégies de retrieval configurables dynamiquement.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ..core.interfaces import IRetriever, IReranker, IVectorStore, IEmbedder
from ..core.models import Document, Query


class RetrievalMode(Enum):
    """Modes de retrieval disponibles."""
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """
    Configuration complète d'une stratégie de retrieval.
    
    Permet de définir et de changer dynamiquement la stratégie
    sans modification de code.
    """
    # Mode de retrieval
    mode: RetrievalMode = RetrievalMode.DENSE
    
    # Paramètres Dense Retrieval
    dense_top_k: int = 10
    dense_similarity_threshold: Optional[float] = None
    dense_normalize_scores: bool = True
    
    # Paramètres BM25
    bm25_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_min_score: float = 0.0
    
    # Paramètres Hybrid
    hybrid_dense_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5
    hybrid_fusion_strategy: str = "weighted_sum"  # 'weighted_sum', 'rrf', 'max'
    hybrid_rrf_k: int = 60
    hybrid_top_k_per_retriever: int = 20
    hybrid_top_k: int = 10
    
    # Reranking
    enable_reranking: bool = False
    reranker_type: str = "cross-encoder"  # 'cross-encoder', 'cohere', 'noop'
    reranker_top_k: Optional[int] = None
    reranker_min_score: Optional[float] = None
    reranker_model: Optional[str] = None
    
    # Filtres de métadonnées
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Paramètres avancés
    cache_embeddings: bool = True
    use_query_expansion: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la config en dictionnaire."""
        return {
            'mode': self.mode.value,
            'dense': {
                'top_k': self.dense_top_k,
                'similarity_threshold': self.dense_similarity_threshold,
                'normalize_scores': self.dense_normalize_scores
            },
            'bm25': {
                'top_k': self.bm25_top_k,
                'k1': self.bm25_k1,
                'b': self.bm25_b,
                'min_score': self.bm25_min_score
            },
            'hybrid': {
                'dense_weight': self.hybrid_dense_weight,
                'bm25_weight': self.hybrid_bm25_weight,
                'fusion_strategy': self.hybrid_fusion_strategy,
                'rrf_k': self.hybrid_rrf_k,
                'top_k_per_retriever': self.hybrid_top_k_per_retriever,
                'top_k': self.hybrid_top_k
            },
            'reranking': {
                'enabled': self.enable_reranking,
                'type': self.reranker_type,
                'top_k': self.reranker_top_k,
                'min_score': self.reranker_min_score,
                'model': self.reranker_model
            },
            'metadata_filters': self.metadata_filters
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RetrievalConfig':
        """Crée une config depuis un dictionnaire."""
        mode = RetrievalMode(config.get('mode', 'dense'))
        
        dense = config.get('dense', {})
        bm25 = config.get('bm25', {})
        hybrid = config.get('hybrid', {})
        reranking = config.get('reranking', {})
        
        return cls(
            mode=mode,
            dense_top_k=dense.get('top_k', 10),
            dense_similarity_threshold=dense.get('similarity_threshold'),
            dense_normalize_scores=dense.get('normalize_scores', True),
            bm25_top_k=bm25.get('top_k', 10),
            bm25_k1=bm25.get('k1', 1.5),
            bm25_b=bm25.get('b', 0.75),
            bm25_min_score=bm25.get('min_score', 0.0),
            hybrid_dense_weight=hybrid.get('dense_weight', 0.5),
            hybrid_bm25_weight=hybrid.get('bm25_weight', 0.5),
            hybrid_fusion_strategy=hybrid.get('fusion_strategy', 'weighted_sum'),
            hybrid_rrf_k=hybrid.get('rrf_k', 60),
            hybrid_top_k_per_retriever=hybrid.get('top_k_per_retriever', 20),
            hybrid_top_k=hybrid.get('top_k', 10),
            enable_reranking=reranking.get('enabled', False),
            reranker_type=reranking.get('type', 'cross-encoder'),
            reranker_top_k=reranking.get('top_k'),
            reranker_min_score=reranking.get('min_score'),
            reranker_model=reranking.get('model'),
            metadata_filters=config.get('metadata_filters', {})
        )


class RetrievalStrategy(IRetriever):
    """
    Stratégie de retrieval configurable dynamiquement.
    
    Permet de changer de mode, de paramètres, et d'activer/désactiver
    le reranking à la volée.
    """

    def __init__(
        self,
        vector_store: Optional[IVectorStore] = None,
        embedder: Optional[IEmbedder] = None,
        documents: Optional[List[Document]] = None,
        config: Optional[RetrievalConfig] = None,
        **kwargs  # Capture 'search_type', 'score_threshold', 'fetch_k', etc.
    ):
        """
        Initialise la stratégie de retrieval.

        Args:
            vector_store: Vector store pour dense retrieval
            embedder: Embedder pour dense retrieval
            documents: Documents pour BM25
            config: Configuration de retrieval
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.documents = documents or []
        self.config = config or RetrievalConfig()

        # On mappe les arguments "flat" venant du YAML vers la structure RetrievalConfig
        if 'search_type' in kwargs:
            # Exemple: mapper 'similarity' vers le mode DENSE
            if kwargs['search_type'] == 'similarity':
                self.config.mode = RetrievalMode.DENSE
        
        if 'score_threshold' in kwargs:
            self.config.dense_similarity_threshold = kwargs['score_threshold']
            
        if 'top_k' in kwargs:
            self.config.dense_top_k = kwargs['top_k']
        
        # Composants (créés à la demande)
        self._dense_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._reranker = None
    
    def _get_dense_retriever(self) -> IRetriever:
        """Crée ou récupère le dense retriever."""
        if self._dense_retriever is None:
            from .dense_retriever import DenseRetriever, DenseRetrieverConfig
            
            if self.vector_store is None or self.embedder is None:
                raise ValueError(
                    "vector_store and embedder required for dense retrieval"
                )
            
            dense_config = DenseRetrieverConfig(
                top_k=self.config.dense_top_k,
                similarity_threshold=self.config.dense_similarity_threshold,
                normalize_scores=self.config.dense_normalize_scores
            )
            
            self._dense_retriever = DenseRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                config=dense_config
            )
        
        return self._dense_retriever
    
    def _get_bm25_retriever(self) -> IRetriever:
        """Crée ou récupère le BM25 retriever."""
        if self._bm25_retriever is None:
            from .bm25_retriever import BM25Retriever, BM25RetrieverConfig
        
            if not self.documents:
                raise ValueError("documents required for BM25 retrieval")
        
            bm25_config = BM25RetrieverConfig(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
                top_k=self.config.bm25_top_k,
                min_score=self.config.bm25_min_score
            )
        
            self._bm25_retriever = BM25Retriever(
                documents=self.documents,
                config=bm25_config
            )
        
        return self._bm25_retriever
    
    def _get_hybrid_retriever(self) -> IRetriever:
        """Crée ou récupère le hybrid retriever."""
        if self._hybrid_retriever is None:
            from .hybrid_retriever import HybridRetriever, HybridRetrieverConfig
            
            dense = self._get_dense_retriever()
            bm25 = self._get_bm25_retriever()
            
            hybrid_config = HybridRetrieverConfig(
                dense_weight=self.config.hybrid_dense_weight,
                bm25_weight=self.config.hybrid_bm25_weight,
                fusion_strategy=self.config.hybrid_fusion_strategy,
                rrf_k=self.config.hybrid_rrf_k,
                top_k_per_retriever=self.config.hybrid_top_k_per_retriever,
                top_k=self.config.hybrid_top_k
            )
            
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=dense,
                bm25_retriever=bm25,
                config=hybrid_config
            )
        
        return self._hybrid_retriever
    
    def _get_reranker(self) -> Optional[IReranker]:
        """Crée ou récupère le reranker."""
        if not self.config.enable_reranking:
            return None
        
        if self._reranker is None:
            from .reranker import (
                CrossEncoderReranker,
                CohereReranker,
                NoOpReranker,
                RerankerConfig
            )
            
            reranker_config = RerankerConfig(
                top_k=self.config.reranker_top_k,
                min_score=self.config.reranker_min_score
            )
            
            if self.config.reranker_type == "cross-encoder":
                model = (
                    self.config.reranker_model or
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self._reranker = CrossEncoderReranker(
                    model_name=model,
                    config=reranker_config
                )
            elif self.config.reranker_type == "cohere":
                # Nécessite une clé API (à fournir via env ou config)
                import os
                api_key = os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "COHERE_API_KEY environment variable required"
                    )
                model = (
                    self.config.reranker_model or
                    "rerank-multilingual-v3.0"
                )
                self._reranker = CohereReranker(
                    api_key=api_key,
                    model=model,
                    config=reranker_config
                )
            elif self.config.reranker_type == "noop":
                self._reranker = NoOpReranker(config=reranker_config)
            else:
                raise ValueError(
                    f"Unknown reranker type: {self.config.reranker_type}"
                )
        
        return self._reranker
    
    def retrieve(
        self,
        query: Query,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Récupère les documents selon la stratégie configurée.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de documents (override config)
            filters: Filtres de métadonnées
            
        Returns:
            Documents récupérés et éventuellement rerankés
        """
        # Fusionner les filtres
        all_filters = {**self.config.metadata_filters}
        if filters:
            all_filters.update(filters)
        fltrs = getattr(query, "filters", None)
        if fltrs:
            all_filters.update(fltrs)

        # Sélectionner le retriever selon le mode
        if self.config.mode == RetrievalMode.DENSE:
            retriever = self._get_dense_retriever()
        elif self.config.mode == RetrievalMode.BM25:
            retriever = self._get_bm25_retriever()
        elif self.config.mode == RetrievalMode.HYBRID:
            retriever = self._get_hybrid_retriever()
        else:
            raise ValueError(f"Unknown retrieval mode: {self.config.mode}")

        # Récupérer les documents
        documents = retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=all_filters if all_filters else None
        )

        # Appliquer le reranking si activé
        if self.config.enable_reranking and documents:
            reranker = self._get_reranker()
            if reranker:
                documents = reranker.rerank(query, documents, top_k=top_k)

        return documents
    
    def update_config(self, config: RetrievalConfig) -> None:
        """
        Met à jour la configuration dynamiquement.
        
        Args:
            config: Nouvelle configuration
        """
        self.config = config
        
        # Réinitialiser les composants pour recréation
        self._dense_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._reranker = None
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return self.config.to_dict()


def create_retriever(
    mode: str = "dense",
    vector_store: Optional[IVectorStore] = None,
    embedder: Optional[IEmbedder] = None,
    documents: Optional[List[Document]] = None,
    config: Optional[Dict[str, Any]] = None
) -> RetrievalStrategy:
    """
    Factory function pour créer une stratégie de retrieval.
    
    Args:
        mode: Mode de retrieval ('dense', 'bm25', 'hybrid')
        vector_store: Vector store pour dense retrieval
        embedder: Embedder pour dense retrieval
        documents: Documents pour BM25
        config: Configuration personnalisée (dict)
        
    Returns:
        Stratégie de retrieval configurée
    """
    # Créer la config
    if config:
        retrieval_config = RetrievalConfig.from_dict(config)
    else:
        retrieval_config = RetrievalConfig(mode=RetrievalMode(mode))
    
    # Créer la stratégie
    return RetrievalStrategy(
        vector_store=vector_store,
        embedder=embedder,
        documents=documents,
        config=retrieval_config
    )
