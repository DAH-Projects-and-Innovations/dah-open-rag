# src/core/interfaces.py
# ==========================================

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Document, Chunk, Query


class IDocumentLoader(ABC):
    """Interface pour charger des documents depuis différentes sources"""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> List[Document]:
        """
        Charge des documents depuis une source
        
        Args:
            source: Chemin, URL ou identifiant de la source
            **kwargs: Paramètres spécifiques au loader
            
        Returns:
            Liste de documents
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Retourne les formats supportés par ce loader"""
        pass


class IChunker(ABC):
    """Interface pour découper des documents en chunks"""
    
    @abstractmethod
    def chunk(self, documents: List[Document], **kwargs) -> List[Chunk]:
        """
        Découpe les documents en chunks
        
        Args:
            documents: Liste de documents à découper
            **kwargs: Paramètres de chunking (taille, overlap, etc.)
            
        Returns:
            Liste de chunks
        """
        pass


class IEmbedder(ABC):
    """Interface pour générer des embeddings"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes
        
        Args:
            texts: Liste de textes
            **kwargs: Paramètres du modèle d'embedding
            
        Returns:
            Liste d'embeddings (vecteurs)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str, **kwargs) -> List[float]:
        """
        Génère un embedding pour une requête
        
        Args:
            query: Texte de la requête
            
        Returns:
            Vecteur d'embedding
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Retourne la dimension des vecteurs"""
        pass


class IVectorStore(ABC):
    """Interface pour stocker et rechercher des vecteurs"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Ajoute des chunks dans le vector store
        
        Args:
            chunks: Liste de chunks avec embeddings
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Chunk]:
        """
        Recherche les chunks les plus similaires
        
        Args:
            query_embedding: Vecteur de la requête
            top_k: Nombre de résultats à retourner
            **kwargs: Paramètres de recherche (filtres, etc.)
            
        Returns:
            Liste de chunks ordonnés par similarité
        """
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Supprime une collection"""
        pass
    
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Retourne les statistiques d'une collection"""
        pass


class IRetriever(ABC):
    """Interface pour la récupération de documents"""
    
    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Récupère les documents pertinents pour une requête
        
        Args:
            query: Requête avec embedding
            top_k: Nombre de documents à retourner
            **kwargs: Paramètres de récupération
            
        Returns:
            Liste de documents pertinents
        """
        pass


class IReranker(ABC):
    """Interface pour réordonner les résultats"""
    
    @abstractmethod
    def rerank(self, query: Query, documents: List[Document], top_k: int = 5, **kwargs) -> List[Document]:
        """
        Réordonne les documents par pertinence

        Args:
            query: Objet requête
            documents: Documents à réordonner
            top_k: Nombre de documents à retourner après reranking
            **kwargs: Paramètres du reranker

        Returns:
            Documents réordonnés
        """
        pass


class IQueryRewriter(ABC):
    """Interface pour réécrire/améliorer les requêtes"""
    
    @abstractmethod
    def rewrite(self, query: str, **kwargs) -> List[str]:
        """
        Réécrit ou génère des variantes d'une requête
        
        Args:
            query: Requête originale
            **kwargs: Paramètres de réécriture
            
        Returns:
            Liste de requêtes réécrites/augmentées
        """
        pass


class ILLM(ABC):
    """Interface pour les modèles de langage"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt
        
        Args:
            prompt: Prompt d'entrée
            **kwargs: Paramètres de génération (temperature, max_tokens, etc.)
            
        Returns:
            Texte généré
        """
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context: List[Document], **kwargs) -> str:
        """
        Génère une réponse en utilisant le contexte
        
        Args:
            query: Question de l'utilisateur
            context: Documents de contexte
            **kwargs: Paramètres de génération
            
        Returns:
            Réponse générée
        """
        pass

