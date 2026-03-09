"""
Tests comparatifs des stratégies de retrieval.

Compare les performances de:
- Dense retrieval
- BM25 retrieval
- Hybrid retrieval (différentes stratégies de fusion)
- Avec et sans reranking
"""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics

from src.core.models import Document, Query
from src.retrieval import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    CrossEncoderReranker,
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalMode
)


@dataclass
class RetrievalMetrics:
    """Métriques d'évaluation du retrieval."""
    strategy_name: str
    avg_latency_ms: float
    precision_at_k: Dict[int, float]  # @1, @3, @5, @10
    recall_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # Normalized Discounted Cumulative Gain


class RetrievalComparator:
    """
    Compare différentes stratégies de retrieval.
    
    Permet d'évaluer et de comparer objectivement les performances
    de différentes configurations.
    """
    
    def __init__(
        self,
        queries: List[Query],
        ground_truth: Dict[str, List[str]],  # query_id -> relevant_doc_ids
        strategies: Dict[str, RetrievalStrategy]
    ):
        """
        Initialise le comparateur.
        
        Args:
            queries: Liste de requêtes de test
            ground_truth: Vérité terrain (docs pertinents par requête)
            strategies: Stratégies à comparer (nom -> strategy)
        """
        self.queries = queries
        self.ground_truth = ground_truth
        self.strategies = strategies
    
    def run_comparison(
        self,
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, RetrievalMetrics]:
        """
        Exécute la comparaison complète.
        
        Args:
            top_k_values: Valeurs de k pour les métriques @k
            
        Returns:
            Métriques par stratégie
        """
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            print(f"\n{'='*60}")
            print(f"Testing strategy: {strategy_name}")
            print(f"{'='*60}")
            
            metrics = self._evaluate_strategy(
                strategy_name,
                strategy,
                top_k_values
            )
            results[strategy_name] = metrics
            
            # Afficher les résultats
            self._print_metrics(metrics)
        
        return results
    
    def _evaluate_strategy(
        self,
        name: str,
        strategy: RetrievalStrategy,
        top_k_values: List[int]
    ) -> RetrievalMetrics:
        """
        Évalue une stratégie de retrieval.
        
        Args:
            name: Nom de la stratégie
            strategy: Stratégie à évaluer
            top_k_values: Valeurs de k à évaluer
            
        Returns:
            Métriques de la stratégie
        """
        latencies = []
        precision_at_k = {k: [] for k in top_k_values}
        recall_at_k = {k: [] for k in top_k_values}
        reciprocal_ranks = []
        ndcg_at_k = {k: [] for k in top_k_values}
        
        for query in self.queries:
            # Mesurer la latence
            start_time = time.time()
            results = strategy.retrieve(query, top_k=max(top_k_values))
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
            # Récupérer la vérité terrain
            relevant_docs = set(self.ground_truth.get(query.id, []))
            if not relevant_docs:
                continue
            
            # Extraire les IDs des documents récupérés
            retrieved_ids = [doc.id for doc in results]
            
            # Calculer les métriques pour chaque k
            for k in top_k_values:
                top_k_ids = retrieved_ids[:k]
                
                # Precision@k
                relevant_at_k = len(set(top_k_ids) & relevant_docs)
                precision = relevant_at_k / k if k > 0 else 0
                precision_at_k[k].append(precision)
                
                # Recall@k
                recall = (
                    relevant_at_k / len(relevant_docs)
                    if len(relevant_docs) > 0 else 0
                )
                recall_at_k[k].append(recall)
                
                # NDCG@k
                ndcg = self._compute_ndcg(top_k_ids, relevant_docs, k)
                ndcg_at_k[k].append(ndcg)
            
            # MRR (Mean Reciprocal Rank)
            rr = self._compute_reciprocal_rank(retrieved_ids, relevant_docs)
            reciprocal_ranks.append(rr)
        
        # Calculer les moyennes
        return RetrievalMetrics(
            strategy_name=name,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            precision_at_k={
                k: statistics.mean(v) if v else 0
                for k, v in precision_at_k.items()
            },
            recall_at_k={
                k: statistics.mean(v) if v else 0
                for k, v in recall_at_k.items()
            },
            mrr=statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0,
            ndcg_at_k={
                k: statistics.mean(v) if v else 0
                for k, v in ndcg_at_k.items()
            }
        )
    
    def _compute_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: set
    ) -> float:
        """
        Calcule le Reciprocal Rank.
        
        Args:
            retrieved: Liste ordonnée des IDs récupérés
            relevant: Set des IDs pertinents
            
        Returns:
            Reciprocal rank (0 si aucun pertinent trouvé)
        """
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    def _compute_ndcg(
        self,
        retrieved: List[str],
        relevant: set,
        k: int
    ) -> float:
        """
        Calcule le Normalized Discounted Cumulative Gain @k.
        
        Args:
            retrieved: Liste ordonnée des IDs récupérés
            relevant: Set des IDs pertinents
            k: Rang jusqu'auquel calculer
            
        Returns:
            NDCG@k score
        """
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            relevance = 1.0 if doc_id in relevant else 0.0
            dcg += relevance / (1 + i)  # log2(1+i) simplifié
        
        # IDCG (Ideal DCG) - si tous les docs pertinents étaient en tête
        num_relevant = min(len(relevant), k)
        idcg = sum(1.0 / (1 + i) for i in range(num_relevant))
        
        # NDCG normalisé
        return dcg / idcg if idcg > 0 else 0.0
    
    def _print_metrics(self, metrics: RetrievalMetrics) -> None:
        """Affiche les métriques de manière lisible."""
        print(f"\nStrategy: {metrics.strategy_name}")
        print(f"Average Latency: {metrics.avg_latency_ms:.2f} ms")
        print(f"MRR: {metrics.mrr:.4f}")
        
        print("\nPrecision@k:")
        for k, v in sorted(metrics.precision_at_k.items()):
            print(f"  @{k}: {v:.4f}")
        
        print("\nRecall@k:")
        for k, v in sorted(metrics.recall_at_k.items()):
            print(f"  @{k}: {v:.4f}")
        
        print("\nNDCG@k:")
        for k, v in sorted(metrics.ndcg_at_k.items()):
            print(f"  @{k}: {v:.4f}")
    
    def print_comparison_table(
        self,
        results: Dict[str, RetrievalMetrics]
    ) -> None:
        """
        Affiche un tableau comparatif des résultats.
        
        Args:
            results: Résultats par stratégie
        """
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Header
        strategies = list(results.keys())
        print(f"\n{'Metric':<20} " + " ".join(f"{s:<15}" for s in strategies))
        print("-" * 80)
        
        # Latency
        print(f"{'Latency (ms)':<20} " + " ".join(
            f"{results[s].avg_latency_ms:<15.2f}" for s in strategies
        ))
        
        # MRR
        print(f"{'MRR':<20} " + " ".join(
            f"{results[s].mrr:<15.4f}" for s in strategies
        ))
        
        # Precision@k
        for k in [1, 3, 5, 10]:
            print(f"{'Precision@' + str(k):<20} " + " ".join(
                f"{results[s].precision_at_k[k]:<15.4f}" for s in strategies
            ))
        
        # NDCG@k
        for k in [5, 10]:
            print(f"{'NDCG@' + str(k):<20} " + " ".join(
                f"{results[s].ndcg_at_k[k]:<15.4f}" for s in strategies
            ))
        
        print("\n" + "="*80)


def create_synthetic_test_data() -> Tuple[
    List[Document],
    List[Query],
    Dict[str, List[str]]
]:
    """
    Crée des données de test synthétiques.
    
    Returns:
        (documents, queries, ground_truth)
    """
    # Documents de test
    documents = [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"topic": "programming", "language": "python"}
        ),
        Document(
            id="doc2",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"topic": "AI", "subtopic": "ML"}
        ),
        Document(
            id="doc3",
            content="Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
            metadata={"topic": "AI", "subtopic": "deep_learning"}
        ),
        Document(
            id="doc4",
            content="Natural language processing helps computers understand and generate human language.",
            metadata={"topic": "AI", "subtopic": "NLP"}
        ),
        Document(
            id="doc5",
            content="Data science combines statistics, programming, and domain expertise to extract insights from data.",
            metadata={"topic": "data_science"}
        ),
    ]
    
    # Requêtes de test
    queries = [
        Query(id="q1", text="What is Python programming?"),
        Query(id="q2", text="Explain machine learning"),
        Query(id="q3", text="How does deep learning work?"),
        Query(id="q4", text="NLP applications"),
    ]
    
    # Vérité terrain (docs pertinents par requête)
    ground_truth = {
        "q1": ["doc1"],
        "q2": ["doc2", "doc3"],
        "q3": ["doc3", "doc2"],
        "q4": ["doc4"],
    }
    
    return documents, queries, ground_truth


if __name__ == "__main__":
    print("Retrieval Strategy Comparison")
    print("=" * 80)
    
    # Note: Ce script est un exemple de structure
    # Pour l'exécuter, il faut:
    # 1. Implémenter les interfaces IVectorStore et IEmbedder
    # 2. Créer les instances concrètes
    # 3. Initialiser les stratégies
    
    print("\nTo run this comparison:")
    print("1. Implement IVectorStore and IEmbedder")
    print("2. Create concrete instances")
    print("3. Initialize strategies with your data")
    print("4. Call comparator.run_comparison()")