"""
BM25 Retriever - Récupération basée sur BM25 (recherche lexicale).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math
from collections import Counter
import re

from ..core.interfaces import IRetriever
from ..core.models import Query, Document


@dataclass
class BM25RetrieverConfig:
    """Configuration pour BM25."""
    k1: float = 1.5  # Paramètre de saturation des termes
    b: float = 0.75  # Paramètre de normalisation de longueur
    top_k: int = 10
    min_score: float = 0.0
    use_stemming: bool = False
    remove_stopwords: bool = True


class BM25Retriever(IRetriever):
    """
    BM25 Retriever pour recherche lexicale.
    
    Implémente l'algorithme BM25 pour scorer les documents
    basé sur la fréquence des termes et leur rareté.
    """
    
    def __init__(
        self,
        documents: List[Document],
        config: Optional[BM25RetrieverConfig] = None
    ):
        """
        Initialise le BM25 Retriever.
        
        Args:
            documents: Corpus de documents à indexer
            config: Configuration BM25
        """
        self.config = config or BM25RetrieverConfig()
        self.documents = documents
        
        # Stopwords français et anglais (simplifié)
        self.stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
            'à', 'au', 'aux', 'ce', 'ces', 'cette', 'dans', 'pour', 'par',
            'sur', 'avec', 'sans', 'en', 'est', 'sont', 'a', 'ont',
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be'
        }
        
        # Indexation
        self._index_documents()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte.
        
        Args:
            text: Texte à tokeniser
            
        Returns:
            Liste de tokens
        """
        # Nettoyage et tokenisation simple
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Suppression des stopwords si configuré
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def _index_documents(self) -> None:
        """Indexe les documents pour BM25."""
        self.doc_tokens = []
        self.doc_lengths = []
        self.term_doc_freq = Counter()
        
        # Tokeniser tous les documents
        for doc in self.documents:
            tokens = self._tokenize(doc.content)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Compter les documents contenant chaque terme
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.term_doc_freq[token] += 1
        
        # Calculer la longueur moyenne des documents
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0
        )
        
        self.num_docs = len(self.documents)
    
    def _compute_idf(self, term: str) -> float:
        """
        Calcule l'IDF (Inverse Document Frequency) d'un terme.
        
        Args:
            term: Terme à scorer
            
        Returns:
            Score IDF
        """
        doc_freq = self.term_doc_freq.get(term, 0)
        if doc_freq == 0:
            return 0.0
        
        # IDF classique avec smoothing
        idf = math.log(
            (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
        )
        return max(idf, 0.0)
    
    def _compute_bm25_score(
        self,
        query_tokens: List[str],
        doc_idx: int
    ) -> float:
        """
        Calcule le score BM25 pour un document.
        
        Args:
            query_tokens: Tokens de la requête
            doc_idx: Index du document
            
        Returns:
            Score BM25
        """
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Comptage des termes dans le document
        term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            # Fréquence du terme dans le document
            tf = term_freqs[term]
            
            # IDF du terme
            idf = self._compute_idf(term)
            
            # Normalisation de longueur
            norm_factor = (
                self.config.k1 * (
                    1 - self.config.b +
                    self.config.b * (doc_length / self.avg_doc_length)
                )
            )
            
            # Score BM25 pour ce terme
            term_score = idf * (
                (tf * (self.config.k1 + 1)) /
                (tf + norm_factor)
            )
            
            score += term_score
        
        return score
    
    def retrieve(
        self,
        query: Query,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Récupère les documents les plus pertinents via BM25.
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de documents à récupérer
            filters: Filtres de métadonnées optionnels
            
        Returns:
            Liste de documents ordonnés par pertinence BM25
        """
        k = top_k or self.config.top_k
        
        # Tokeniser la requête
        query_tokens = self._tokenize(query.text)
        
        # Scorer tous les documents
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            # Vérifier les filtres de métadonnées
            if filters and not self._match_filters(doc, filters):
                continue
            query_filters = getattr(query, 'filters', None)
            if query_filters and not self._match_filters(doc, query_filters):
                continue
            
            score = self._compute_bm25_score(query_tokens, idx)
            
            # Filtrer par score minimum
            if score < self.config.min_score:
                continue
            
            # Créer une copie du document avec le score
            doc_copy = Document(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata={
                    **doc.metadata,
                    'bm25_score': score,
                    'score': score
                }
            )
            scored_docs.append((score, doc_copy))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Retourner top-k
        return [doc for _, doc in scored_docs[:k]]
    
    def _match_filters(
        self,
        document: Document,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Vérifie si un document correspond aux filtres.
        
        Args:
            document: Document à vérifier
            filters: Filtres à appliquer
            
        Returns:
            True si le document correspond aux filtres
        """
        for key, value in filters.items():
            doc_value = document.metadata.get(key)
            
            # Support de filtres complexes
            if isinstance(value, dict):
                # Opérateurs: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
                if '$eq' in value and doc_value != value['$eq']:
                    return False
                if '$ne' in value and doc_value == value['$ne']:
                    return False
                if '$gt' in value and not (doc_value > value['$gt']):
                    return False
                if '$gte' in value and not (doc_value >= value['$gte']):
                    return False
                if '$lt' in value and not (doc_value < value['$lt']):
                    return False
                if '$lte' in value and not (doc_value <= value['$lte']):
                    return False
                if '$in' in value and doc_value not in value['$in']:
                    return False
                if '$nin' in value and doc_value in value['$nin']:
                    return False
            else:
                # Comparaison directe
                if doc_value != value:
                    return False
        
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return {
            'type': 'bm25',
            'k1': self.config.k1,
            'b': self.config.b,
            'top_k': self.config.top_k,
            'min_score': self.config.min_score
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
        
        # Ré-indexer si les paramètres affectent l'indexation
        if 'remove_stopwords' in kwargs or 'use_stemming' in kwargs:
            self._index_documents()