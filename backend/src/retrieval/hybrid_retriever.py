"""
Hybrid Retriever - Combine recherche lexicale (BM25) et vectorielle (dense).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..core.interfaces import IRetriever
from ..core.models import Query, Document


@dataclass
class HybridRetrieverConfig:
    """Configuration pour le Hybrid Retriever."""
    # Pondération des scores
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    
    # Fusion strategy: 'weighted_sum', 'rrf' (Reciprocal Rank Fusion), 'max'
    fusion_strategy: str = 'weighted_sum'
    
    # Paramètres RRF
    rrf_k: int = 60
    
    # Nombre de résultats à récupérer de chaque retriever
    top_k_per_retriever: int = 20
    
    # Nombre final de résultats
    top_k: int = 10
    
    # Normalisation des scores avant fusion
    normalize_scores: bool = True


class HybridRetriever(IRetriever):
    """
    Hybrid Retriever combinant BM25 et dense retrieval.
    
    Fusionne les résultats de deux retrievers pour bénéficier
    des avantages de la recherche lexicale et sémantique.
    """
    
    def __init__(
        self,
        dense_retriever: IRetriever,
        bm25_retriever: IRetriever,
        config: Optional[HybridRetrieverConfig] = None
    ):
        """
        Initialise le Hybrid Retriever.
        
        Args:
            dense_retriever: Retriever basé embeddings
            bm25_retriever: Retriever BM25
            config: Configuration du retriever hybride
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridRetrieverConfig()
        
        # Valider les poids
        self._validate_weights()
    
    def _validate_weights(self) -> None:
        """Valide que les poids sont cohérents."""
        total_weight = self.config.dense_weight + self.config.bm25_weight
        if abs(total_weight - 1.0) > 1e-6:
            # Normaliser automatiquement
            self.config.dense_weight /= total_weight
            self.config.bm25_weight /= total_weight
    
    def retrieve(
        self,
        query: Query,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Récupère les documents via retrieval hybride.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de documents à récupérer
            filters: Filtres de métadonnées optionnels
            
        Returns:
            Liste de documents fusionnés et ordonnés
        """
        k = top_k or self.config.top_k
        
        # Récupérer les résultats des deux retrievers
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.config.top_k_per_retriever,
            filters=filters
        )
        
        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=self.config.top_k_per_retriever,
            filters=filters
        )
        
        # Fusionner les résultats selon la stratégie
        if self.config.fusion_strategy == 'weighted_sum':
            fused_results = self._weighted_sum_fusion(
                dense_results, bm25_results
            )
        elif self.config.fusion_strategy == 'rrf':
            fused_results = self._rrf_fusion(
                dense_results, bm25_results
            )
        elif self.config.fusion_strategy == 'max':
            fused_results = self._max_fusion(
                dense_results, bm25_results
            )
        else:
            raise ValueError(
                f"Unknown fusion strategy: {self.config.fusion_strategy}"
            )
        
        # Retourner top-k
        return fused_results[:k]
    
    def _normalize_scores(self, documents: List[Document]) -> List[Document]:
        """
        Normalise les scores entre 0 et 1.
        
        Args:
            documents: Documents à normaliser
            
        Returns:
            Documents avec scores normalisés
        """
        if not documents:
            return documents
        
        scores = [doc.metadata.get('score', 0) for doc in documents]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            for doc in documents:
                doc.metadata['normalized_score'] = 1.0
            return documents
        
        for doc in documents:
            original_score = doc.metadata.get('score', 0)
            normalized = (original_score - min_score) / score_range
            doc.metadata['normalized_score'] = normalized
        
        return documents
    
    def _weighted_sum_fusion(
        self,
        dense_results: List[Document],
        bm25_results: List[Document]
    ) -> List[Document]:
        """
        Fusionne via somme pondérée des scores.
        
        Args:
            dense_results: Résultats du dense retriever
            bm25_results: Résultats du BM25 retriever
            
        Returns:
            Documents fusionnés et triés
        """
        # Normaliser les scores si configuré
        if self.config.normalize_scores:
            dense_results = self._normalize_scores(dense_results)
            bm25_results = self._normalize_scores(bm25_results)
        
        # Créer un dictionnaire doc_id -> (doc, scores)
        doc_scores = defaultdict(lambda: {'doc': None, 'scores': []})
        
        # Ajouter les scores dense
        for doc in dense_results:
            score = doc.metadata.get(
                'normalized_score' if self.config.normalize_scores else 'score',
                0
            )
            doc_scores[doc.id]['doc'] = doc
            doc_scores[doc.id]['scores'].append(
                ('dense', score * self.config.dense_weight)
            )
        
        # Ajouter les scores BM25
        for doc in bm25_results:
            score = doc.metadata.get(
                'normalized_score' if self.config.normalize_scores else 'bm25_score',
                0
            )
            if doc_scores[doc.id]['doc'] is None:
                doc_scores[doc.id]['doc'] = doc
            doc_scores[doc.id]['scores'].append(
                ('bm25', score * self.config.bm25_weight)
            )
        
        # Calculer les scores finaux
        fused_docs = []
        for doc_id, data in doc_scores.items():
            doc = data['doc']
            
            # Somme des scores pondérés
            final_score = sum(score for _, score in data['scores'])
            
            # Mettre à jour les métadonnées
            doc.metadata['hybrid_score'] = final_score
            doc.metadata['fusion_details'] = {
                name: score for name, score in data['scores']
            }
            
            fused_docs.append((final_score, doc))
        
        # Trier par score décroissant
        fused_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in fused_docs]
    
    def _rrf_fusion(
        self,
        dense_results: List[Document],
        bm25_results: List[Document]
    ) -> List[Document]:
        """
        Fusionne via Reciprocal Rank Fusion (RRF).
        
        RRF est robuste aux différences d'échelles de scores.
        Score RRF = sum(1 / (k + rank)) pour chaque retriever.
        
        Args:
            dense_results: Résultats du dense retriever
            bm25_results: Résultats du BM25 retriever
            
        Returns:
            Documents fusionnés et triés
        """
        k = self.config.rrf_k
        doc_scores = defaultdict(lambda: {'doc': None, 'rrf_score': 0.0})
        
        # Ajouter les scores RRF du dense retriever
        for rank, doc in enumerate(dense_results, start=1):
            rrf_contribution = 1.0 / (k + rank)
            doc_scores[doc.id]['doc'] = doc
            doc_scores[doc.id]['rrf_score'] += (
                rrf_contribution * self.config.dense_weight
            )
        
        # Ajouter les scores RRF du BM25 retriever
        for rank, doc in enumerate(bm25_results, start=1):
            rrf_contribution = 1.0 / (k + rank)
            if doc_scores[doc.id]['doc'] is None:
                doc_scores[doc.id]['doc'] = doc
            doc_scores[doc.id]['rrf_score'] += (
                rrf_contribution * self.config.bm25_weight
            )
        
        # Créer la liste fusionnée
        fused_docs = []
        for doc_id, data in doc_scores.items():
            doc = data['doc']
            doc.metadata['rrf_score'] = data['rrf_score']
            doc.metadata['hybrid_score'] = data['rrf_score']
            fused_docs.append((data['rrf_score'], doc))
        
        # Trier par score RRF décroissant
        fused_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in fused_docs]
    
    def _max_fusion(
        self,
        dense_results: List[Document],
        bm25_results: List[Document]
    ) -> List[Document]:
        """
        Fusionne en prenant le max des scores normalisés.
        
        Args:
            dense_results: Résultats du dense retriever
            bm25_results: Résultats du BM25 retriever
            
        Returns:
            Documents fusionnés et triés
        """
        # Normaliser les scores
        dense_results = self._normalize_scores(dense_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        doc_scores = {}
        
        # Collecter les scores
        for doc in dense_results:
            score = doc.metadata.get('normalized_score', 0)
            doc_scores[doc.id] = {
                'doc': doc,
                'dense_score': score,
                'bm25_score': 0
            }
        
        for doc in bm25_results:
            score = doc.metadata.get('normalized_score', 0)
            if doc.id in doc_scores:
                doc_scores[doc.id]['bm25_score'] = score
            else:
                doc_scores[doc.id] = {
                    'doc': doc,
                    'dense_score': 0,
                    'bm25_score': score
                }
        
        # Calculer le max des scores
        fused_docs = []
        for doc_id, data in doc_scores.items():
            doc = data['doc']
            max_score = max(data['dense_score'], data['bm25_score'])
            doc.metadata['hybrid_score'] = max_score
            doc.metadata['max_fusion_details'] = {
                'dense': data['dense_score'],
                'bm25': data['bm25_score']
            }
            fused_docs.append((max_score, doc))
        
        # Trier par score décroissant
        fused_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in fused_docs]
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return {
            'type': 'hybrid',
            'dense_weight': self.config.dense_weight,
            'bm25_weight': self.config.bm25_weight,
            'fusion_strategy': self.config.fusion_strategy,
            'rrf_k': self.config.rrf_k,
            'top_k': self.config.top_k,
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
        
        # Revalider les poids si modifiés
        if 'dense_weight' in kwargs or 'bm25_weight' in kwargs:
            self._validate_weights()
