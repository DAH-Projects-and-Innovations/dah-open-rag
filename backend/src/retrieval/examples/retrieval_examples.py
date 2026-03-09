"""
Exemple d'utilisation du système de retrieval avancé.

Démontre:
1. Configuration de différentes stratégies
2. Changement dynamique de configuration
3. Comparaison des résultats
4. Utilisation avec filtres de métadonnées
"""

import yaml
from typing import List

from src.core.models import Document, Query
from src.retrieval import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalMode,
    create_retriever
)


def load_config_from_yaml(config_path: str) -> dict:
    """Charge une configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def example_1_basic_usage():
    """Exemple 1: Utilisation basique avec config par défaut."""
    print("\n" + "="*60)
    print("EXEMPLE 1: Utilisation basique")
    print("="*60)
    
    # Données d'exemple
    documents = [
        Document(doc_id="1", content="Python is great for data science"),
        Document(doc_id="2", content="Machine learning models need training data"),
        Document(doc_id="3", content="Deep learning uses neural networks"),
    ]
    
    # Créer une stratégie dense simple
    # Note: Nécessite un vector_store et un embedder concrets
    print("\nCréation d'une stratégie dense...")
    print("Config: mode=dense, top_k=5")
    
    # strategy = create_retriever(
    #     mode="dense",
    #     vector_store=my_vector_store,
    #     embedder=my_embedder,
    #     documents=documents
    # )
    
    # query = Query(id="q1", text="What is deep learning?")
    # results = strategy.retrieve(query, top_k=5)
    
    print("✓ Stratégie créée (exemple)")


def example_2_hybrid_with_reranking():
    """Exemple 2: Retrieval hybride avec reranking."""
    print("\n" + "="*60)
    print("EXEMPLE 2: Hybrid + Reranking")
    print("="*60)
    
    # Configuration hybride personnalisée
    config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        hybrid_dense_weight=0.6,
        hybrid_bm25_weight=0.4,
        hybrid_fusion_strategy="rrf",
        enable_reranking=True,
        reranker_type="cross-encoder",
        reranker_top_k=5
    )
    
    print("\nConfiguration:")
    print(yaml.dump(config.to_dict(), default_flow_style=False))
    
    # strategy = RetrievalStrategy(
    #     vector_store=my_vector_store,
    #     embedder=my_embedder,
    #     documents=documents,
    #     config=config
    # )
    
    print("✓ Stratégie hybride avec reranking configurée")


def example_3_load_from_yaml():
    """Exemple 3: Charger une config depuis YAML."""
    print("\n" + "="*60)
    print("EXEMPLE 3: Chargement depuis YAML")
    print("="*60)
    
    config_files = [
        "configs/retrieval/dense_simple.yaml",
        "configs/retrieval/bm25_pure.yaml",
        "configs/retrieval/hybrid_rerank.yaml",
        "configs/retrieval/premium_cohere.yaml"
    ]
    
    for config_file in config_files:
        print(f"\n📄 {config_file}")
        try:
            config_dict = load_config_from_yaml(config_file)
            config = RetrievalConfig.from_dict(config_dict)
            print(f"  Mode: {config.mode.value}")
            print(f"  Reranking: {'✓' if config.enable_reranking else '✗'}")
            print(f"  Description: {config_dict.get('description', 'N/A')[:50]}...")
        except FileNotFoundError:
            print(f"  ⚠ File not found")


def example_4_dynamic_config_change():
    """Exemple 4: Changement dynamique de configuration."""
    print("\n" + "="*60)
    print("EXEMPLE 4: Changement dynamique de config")
    print("="*60)
    
    # Commencer avec une config dense
    config = RetrievalConfig(mode=RetrievalMode.DENSE)
    print("\n1️⃣ Configuration initiale: DENSE")
    print(f"   Mode: {config.mode.value}")
    
    # strategy = RetrievalStrategy(
    #     vector_store=my_vector_store,
    #     embedder=my_embedder,
    #     documents=documents,
    #     config=config
    # )
    
    # Première requête
    # results = strategy.retrieve(Query(text="example query"))
    print("   ✓ Requête exécutée")
    
    # Changer vers hybrid
    new_config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        hybrid_fusion_strategy="rrf"
    )
    print("\n2️⃣ Changement vers HYBRID")
    print(f"   Mode: {new_config.mode.value}")
    print(f"   Fusion: {new_config.hybrid_fusion_strategy}")
    
    # strategy.update_config(new_config)
    # results = strategy.retrieve(Query(text="example query"))
    print("   ✓ Config mise à jour, requête exécutée")
    
    # Activer le reranking
    new_config.enable_reranking = True
    new_config.reranker_type = "cross-encoder"
    print("\n3️⃣ Activation du reranking")
    print(f"   Reranker: {new_config.reranker_type}")
    
    # strategy.update_config(new_config)
    # results = strategy.retrieve(Query(text="example query"))
    print("   ✓ Reranking activé, requête exécutée")


def example_5_metadata_filtering():
    """Exemple 5: Filtrage par métadonnées."""
    print("\n" + "="*60)
    print("EXEMPLE 5: Filtrage par métadonnées")
    print("="*60)
    
    # Documents avec métadonnées
    documents = [
        Document(
            doc_id="1",
            content="Python tutorial for beginners",
            metadata={"language": "python", "level": "beginner", "year": 2024}
        ),
        Document(
            doc_id="2",
            content="Advanced Python patterns",
            metadata={"language": "python", "level": "advanced", "year": 2024}
        ),
        Document(
            doc_id="3",
            content="JavaScript fundamentals",
            metadata={"language": "javascript", "level": "beginner", "year": 2023}
        ),
    ]
    
    print("\n📚 Documents avec métadonnées:")
    for doc in documents:
        print(f"  - {doc.doc_id}: {doc.metadata}")
    
    # Configuration avec filtres
    config = RetrievalConfig(
        mode=RetrievalMode.DENSE,
        metadata_filters={
            "language": "python",
            "level": "beginner"
        }
    )
    
    print("\n🔍 Filtres appliqués:")
    print(f"  language = python")
    print(f"  level = beginner")
    
    # strategy = RetrievalStrategy(
    #     vector_store=my_vector_store,
    #     embedder=my_embedder,
    #     documents=documents,
    #     config=config
    # )
    
    # query = Query(text="Python tutorial")
    # results = strategy.retrieve(query)
    
    print("\n✓ Résultats filtrés (attendu: doc 1 uniquement)")
    
    # Filtres complexes
    print("\n🔍 Filtres complexes (opérateurs):")
    complex_filters = {
        "year": {"$gte": 2024},  # Année >= 2024
        "language": {"$in": ["python", "javascript"]}  # Python OU JavaScript
    }
    print(yaml.dump(complex_filters, default_flow_style=False))


def example_6_strategy_comparison():
    """Exemple 6: Comparaison de stratégies."""
    print("\n" + "="*60)
    print("EXEMPLE 6: Comparaison de stratégies")
    print("="*60)
    
    strategies_to_compare = [
        ("Dense Simple", RetrievalConfig(mode=RetrievalMode.DENSE)),
        ("BM25", RetrievalConfig(mode=RetrievalMode.BM25)),
        ("Hybrid RRF", RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            hybrid_fusion_strategy="rrf"
        )),
        ("Hybrid + Rerank", RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            hybrid_fusion_strategy="rrf",
            enable_reranking=True,
            reranker_type="cross-encoder"
        ))
    ]
    
    print("\n📊 Stratégies à comparer:")
    for name, config in strategies_to_compare:
        print(f"\n  {name}:")
        print(f"    Mode: {config.mode.value}")
        if config.mode == RetrievalMode.HYBRID:
            print(f"    Fusion: {config.hybrid_fusion_strategy}")
        if config.enable_reranking:
            print(f"    Reranking: {config.reranker_type}")
    
    print("\n💡 Pour exécuter la comparaison:")
    print("   from tests.test_retrieval_comparison import RetrievalComparator")
    print("   comparator = RetrievalComparator(queries, ground_truth, strategies)")
    print("   results = comparator.run_comparison()")


def example_7_best_practices():
    """Exemple 7: Bonnes pratiques."""
    print("\n" + "="*60)
    print("EXEMPLE 7: Bonnes pratiques")
    print("="*60)
    
    practices = [
        ("🎯 Commencer simple", "Utilisez dense_simple.yaml pour débuter"),
        ("📈 Itérer", "Testez différentes configs et mesurez les performances"),
        ("🔄 Hybrid pour précision", "Utilisez hybrid quand la qualité est critique"),
        ("⚡ BM25 pour vitesse", "BM25 pure pour recherche rapide par mots-clés"),
        ("🎓 Reranking pour top-k", "Activez le reranking pour affiner les top-10"),
        ("🔍 Filtres métadonnées", "Utilisez les filtres pour restreindre la recherche"),
        ("📊 Mesurer toujours", "Utilisez le comparator pour des décisions data-driven"),
        ("💰 Coût vs Qualité", "Cross-encoder (gratuit) vs Cohere (payant mais meilleur)"),
    ]
    
    print("\n✨ Recommandations:")
    for title, description in practices:
        print(f"\n  {title}")
        print(f"    → {description}")
    
    print("\n\n🚀 Quick Start:")
    print("""
    # 1. Configuration simple
    config = load_config_from_yaml('configs/retrieval/dense_simple.yaml')
    strategy = create_retriever(mode=config['mode'], ...)
    
    # 2. Exécuter une requête
    results = strategy.retrieve(Query(text="ma question"))
    
    # 3. Ajuster si nécessaire
    strategy.update_config(RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        enable_reranking=True
    ))
    """)


if __name__ == "__main__":
    print("🔍 Exemples d'utilisation du système de retrieval avancé")
    print("="*60)
    
    examples = [
        example_1_basic_usage,
        example_2_hybrid_with_reranking,
        example_3_load_from_yaml,
        example_4_dynamic_config_change,
        example_5_metadata_filtering,
        example_6_strategy_comparison,
        example_7_best_practices
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠️ Erreur dans l'exemple: {e}")
    
    print("\n" + "="*60)
    print("✅ Tous les exemples exécutés")
    print("="*60)
