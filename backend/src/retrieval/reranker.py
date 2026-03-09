"""
Rerankers - Réordonne les documents par pertinence via des modèles de reranking.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.interfaces import IReranker
from ..core.models import Query, Document


@dataclass
class RerankerConfig:
    """Configuration de base pour les rerankers."""
    top_k: Optional[int] = None  # None = retourner tous les documents
    min_score: Optional[float] = None


class BaseReranker(IReranker, ABC):
    """Classe de base abstraite pour tous les rerankers."""
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialise le reranker.
        
        Args:
            config: Configuration du reranker
        """
        self.config = config or RerankerConfig()
    
    @abstractmethod
    def _compute_relevance_score(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Calcule le score de pertinence entre une requête et un document.
        
        Args:
            query: Texte de la requête
            document: Texte du document
            
        Returns:
            Score de pertinence
        """
        pass
    
    def rerank(
        self,
        query: Query,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Réordonne les documents par pertinence.
        
        Args:
            query: Requête utilisateur
            documents: Documents à réordonner
            top_k: Nombre de documents à retourner (override config)
            
        Returns:
            Documents réordonnés par pertinence
        """
        if not documents:
            return documents
        
        # Calculer les scores de pertinence
        scored_docs = []
        for doc in documents:
            score = self._compute_relevance_score(query.text, doc.content)
            
            # Filtrer par score minimum si configuré
            if self.config.min_score is not None:
                if score < self.config.min_score:
                    continue
            
            # Mettre à jour les métadonnées
            doc.metadata['rerank_score'] = score
            doc.metadata['original_score'] = doc.metadata.get('score', 0)
            doc.metadata['score'] = score  # Remplacer le score principal
            
            scored_docs.append((score, doc))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Appliquer top-k
        k = top_k or self.config.top_k
        if k is not None:
            scored_docs = scored_docs[:k]
        
        return [doc for _, doc in scored_docs]
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return {
            'top_k': self.config.top_k,
            'min_score': self.config.min_score
        }


class NoOpReranker(BaseReranker):
    """
    Reranker qui ne fait rien (pass-through).
    
    Utile pour désactiver le reranking dans un pipeline configuré.
    """
    
    def _compute_relevance_score(self, query: str, document: str) -> float:
        """Retourne toujours 1.0 (pas de reranking)."""
        return 1.0
    
    def rerank(
        self,
        query: Query,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Retourne les documents sans modification."""
        return documents


class CrossEncoderReranker(BaseReranker):
    """
    Reranker basé sur un cross-encoder (Sentence Transformers).
    
    Utilise un modèle type 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    qui encode la paire (query, document) ensemble.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size:int = 16,
        config: Optional[RerankerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise le CrossEncoder reranker.
        
        Args:
            model_name: Nom du modèle HuggingFace
            config: Configuration du reranker
            device: Device pour le modèle ('cuda', 'cpu', None=auto)
        """
        super().__init__(config)
        self.model_name = model_name
        self.device = device
        self._model = None
        self.batch_size = batch_size
    
    def _load_model(self):
        """Charge le modèle de manière lazy."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )
    
    def _compute_relevance_score(self, query: str, document: str) -> float:
        """
        Calcule le score de pertinence via cross-encoder.
        
        Args:
            query: Texte de la requête
            document: Texte du document
            
        Returns:
            Score de pertinence
        """
        self._load_model()
        
        # Le cross-encoder prend une paire (query, document)
        score = self._model.predict([(query, document)])[0]
        
        # Convertir en float standard
        return float(score)
    
    def rerank(
        self,
        query: Query,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Réordonne via cross-encoder (optimisé pour batch).
        
        Args:
            query: Requête utilisateur
            documents: Documents à réordonner
            top_k: Nombre de documents à retourner
            
        Returns:
            Documents réordonnés
        """
        if not documents:
            return documents
        
        self._load_model()
        
        # Créer les paires (query, document) pour batch scoring
        pairs = [(query.text, doc.content) for doc in documents]
        
        # Scorer en batch (plus efficace)
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        
        # Associer scores et documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            score = float(score)
            
            # Filtrer par score minimum
            if self.config.min_score is not None:
                if score < self.config.min_score:
                    continue
            
            # Mettre à jour métadonnées
            doc.metadata['rerank_score'] = score
            doc.metadata['original_score'] = doc.metadata.get('score', 0)
            doc.metadata['score'] = score
            
            scored_docs.append((score, doc))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Appliquer top-k
        k = top_k or self.config.top_k
        if k is not None:
            scored_docs = scored_docs[:k]
        
        return [doc for _, doc in scored_docs]
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        config = super().get_config()
        config['model_name'] = self.model_name
        config['device'] = self.device
        return config


class CohereReranker(BaseReranker):
    """
    Reranker utilisant l'API Cohere Rerank.
    
    Utilise le modèle 'rerank-multilingual-v3.0' ou similaire.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
        config: Optional[RerankerConfig] = None
    ):
        """
        Initialise le Cohere reranker.
        
        Args:
            api_key: Clé API Cohere
            model: Nom du modèle de reranking
            config: Configuration du reranker
        """
        super().__init__(config)
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _load_client(self):
        """Charge le client Cohere de manière lazy."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError(
                    "cohere is required for CohereReranker. "
                    "Install with: pip install cohere"
                )
    
    def _compute_relevance_score(self, query: str, document: str) -> float:
        """
        Compute relevance via Cohere API (utilisé par la méthode de base).
        
        Note: Cette méthode est moins efficace que rerank() en batch.
        """
        self._load_client()
        
        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=[document],
            top_n=1
        )
        
        if response.results:
            return response.results[0].relevance_score
        return 0.0
    
    def rerank(
        self,
        query: Query,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Réordonne via Cohere API (optimisé pour batch).
        
        Args:
            query: Requête utilisateur
            documents: Documents à réordonner
            top_k: Nombre de documents à retourner
            
        Returns:
            Documents réordonnés
        """
        if not documents:
            return documents
        
        self._load_client()
        
        # Préparer les documents pour Cohere
        doc_texts = [doc.content for doc in documents]
        
        # Appeler l'API Cohere en batch
        k = top_k or self.config.top_k or len(documents)
        response = self._client.rerank(
            model=self.model,
            query=query.text,
            documents=doc_texts,
            top_n=k
        )
        
        # Créer un mapping index -> score
        scores_map = {}
        for result in response.results:
            scores_map[result.index] = result.relevance_score
        
        # Réordonner les documents
        scored_docs = []
        for idx, doc in enumerate(documents):
            if idx not in scores_map:
                continue  # Document pas dans top-k de Cohere
            
            score = scores_map[idx]
            
            # Filtrer par score minimum
            if self.config.min_score is not None:
                if score < self.config.min_score:
                    continue
            
            # Mettre à jour métadonnées
            doc.metadata['rerank_score'] = score
            doc.metadata['original_score'] = doc.metadata.get('score', 0)
            doc.metadata['score'] = score
            
            scored_docs.append((score, doc))
        
        # Trier par score décroissant (normalement déjà trié par Cohere)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs]
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        config = super().get_config()
        config['model'] = self.model
        config['api_key'] = '***'  # Masquer la clé
        return config
        