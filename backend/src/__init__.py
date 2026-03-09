# ==========================================
# src/__init__.py
# Point d'entrée principal du package
# ==========================================

"""
Architecture RAG Modulaire et Configurable

Ce package fournit une architecture complète pour construire des systèmes
RAG (Retrieval-Augmented Generation) modulaires et configurables.

Usage rapide:
    >>> from src.core import RAGPipelineFactory
    >>> from src.implementations import register_all_components
    >>> 
    >>> register_all_components()
    >>> config = RAGPipelineFactory.load_config('configs/hybrid.yaml')
    >>> pipeline = RAGPipelineFactory.create_from_config(config)
    >>> 
    >>> response = pipeline.query("Quelle est la question?")
    >>> print(response.answer)

Composants principaux:
    - RAGPipeline: Orchestrateur central
    - RAGPipelineFactory: Factory pour créer des pipelines
    - IDocumentLoader, IChunker, IEmbedder, etc.: Interfaces abstraites
"""

__version__ = "1.0.0"
__author__ = "Dobé Nancy"
__license__ = "MIT"

# Core exports
from .core import (
    # Models
    Document,
    Chunk,
    Query,
    RAGResponse,
    # Interfaces
    IDocumentLoader,
    IChunker,
    IEmbedder,
    IVectorStore,
    IRetriever,
    IReranker,
    IQueryRewriter,
    ILLM,
    # Core classes
    RAGPipeline,
    RAGPipelineFactory,
)

__all__ = [
    # Version
    '__version__',
    # Models
    'Document',
    'Chunk',
    'Query',
    'RAGResponse',
    # Interfaces
    'IDocumentLoader',
    'IChunker',
    'IEmbedder',
    'IVectorStore',
    'IRetriever',
    'IReranker',
    'IQueryRewriter',
    'ILLM',
    # Core classes
    'RAGPipeline',
    'RAGPipelineFactory',
]


def get_version():
    """Retourne la version du package"""
    return __version__


def quick_start(config_path: str = 'configs/hybrid.yaml'):
    """
    Démarrage rapide avec configuration par défaut
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Pipeline RAG prêt à l'emploi
        
    Example:
        >>> pipeline = quick_start()
        >>> response = pipeline.query("Test question")
    """
    try:
        from .implementations import register_all_components
        
        # Enregistrer tous les composants
        register_all_components()
        
        # Charger la configuration
        config = RAGPipelineFactory.load_config(config_path)
        
        # Créer le pipeline
        pipeline = RAGPipelineFactory.create_from_config(config)
        
        print(f"✅ Pipeline initialisé avec la configuration: {config_path}")
        return pipeline
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        raise


def list_available_components():
    """
    Liste tous les composants disponibles
    
    Returns:
        Dictionnaire des composants par catégorie
    """
    try:
        from .implementations import register_all_components
        register_all_components()
        return RAGPipelineFactory.list_components()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {}


def setup_logging(level: str = "INFO"):
    """
    Configure le logging pour le package
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
    """
    import logging
    
    # Configuration du logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Logger spécifique pour le package
    logger = logging.getLogger('src')
    logger.setLevel(getattr(logging, level.upper()))
    
    print(f"✅ Logging configuré au niveau: {level}")


# Message de bienvenue lors de l'import
def _welcome_message():
    """Affiche un message de bienvenue"""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║        Architecture RAG Modulaire v{__version__}                  ║
║                                                           ║
║  Démarrage rapide:                                        ║
║    >>> from src import quick_start                        ║
║    >>> pipeline = quick_start()                           ║
║                                                           ║
║  Documentation: voir README.md                            ║
╚═══════════════════════════════════════════════════════════╝
    """)


# Afficher le message seulement si pas en mode import silencieux
import os
if os.getenv('RAG_SILENT_IMPORT') != '1':
    _welcome_message()


# ==========================================
# Configuration par défaut
# ==========================================

DEFAULT_CONFIG = {
    'embedder': {
        'name': 'sentence_transformers',
        'params': {
            'model_name': 'BAAI/bge-large-en-v1.5',
            'device': 'cpu'
        }
    },
    'vector_store': {
        'name': 'chroma',
        'params': {
            'collection_name': 'documents',
            'persist_directory': './data/chroma_db'
        }
    },
    'retriever': {
        'name': 'vector_retriever',
        'params': {
            'search_type': 'similarity'
        }
    },
    'llm': {
        'name': 'ollama',
        'params': {
            'model': 'llama3.1:8b',
            'base_url': 'http://localhost:11434'
        }
    },
    'pipeline_config': {
        'default_top_k': 5,
        'enable_caching': True,
        'log_level': 'INFO'
    }
}


def create_default_pipeline():
    """
    Crée un pipeline avec la configuration par défaut
    
    Returns:
        Pipeline RAG avec configuration gratuite
    """
    try:
        from .implementations import register_all_components
        register_all_components()
        
        pipeline = RAGPipelineFactory.create_from_config(DEFAULT_CONFIG)
        print("✅ Pipeline créé avec la configuration par défaut")
        return pipeline
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise


# ==========================================
# Utilitaires
# ==========================================

def validate_environment():
    """
    Vérifie que l'environnement est correctement configuré
    
    Returns:
        Dict avec les résultats de validation
    """
    results = {
        'python_version': True,
        'dependencies': {},
        'env_vars': {},
        'recommendations': []
    }
    
    # Vérifier la version Python
    import sys
    if sys.version_info < (3, 8):
        results['python_version'] = False
        results['recommendations'].append("Python 3.8+ requis")
    
    # Vérifier les dépendances
    required = ['sentence_transformers', 'chromadb', 'yaml', 'fastapi']
    for dep in required:
        try:
            __import__(dep)
            results['dependencies'][dep] = True
        except ImportError:
            results['dependencies'][dep] = False
            results['recommendations'].append(f"Installer {dep}")
    
    # Vérifier les variables d'environnement optionnelles
    optional_env = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'ANTHROPIC_API_KEY']
    for var in optional_env:
        results['env_vars'][var] = bool(os.getenv(var))
    
    return results


def print_validation_report():
    """Affiche un rapport de validation de l'environnement"""
    results = validate_environment()
    
    print("\n📋 Rapport de validation de l'environnement:")
    print(f"  Python: {'✅' if results['python_version'] else '❌'}")
    
    print("\n  Dépendances:")
    for dep, status in results['dependencies'].items():
        print(f"    {dep}: {'✅' if status else '❌'}")
    
    print("\n  Variables d'environnement (optionnelles):")
    for var, status in results['env_vars'].items():
        print(f"    {var}: {'✅' if status else '⚠️  non défini'}")
    
    if results['recommendations']:
        print("\n  💡 Recommandations:")
        for rec in results['recommendations']:
            print(f"    - {rec}")
    else:
        print("\n  ✅ Environnement prêt!")


# ==========================================
# Helpers pour le développement
# ==========================================

def create_sample_documents(output_dir: str = './data/documents'):
    """
    Crée des documents d'exemple pour tester le pipeline
    
    Args:
        output_dir: Répertoire de sortie
    """
    import os
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    samples = {
        'rag_introduction.txt': """
        Le RAG (Retrieval-Augmented Generation) est une technique qui combine
        la recherche d'informations et la génération de texte. Cette approche
        permet aux modèles de langage d'accéder à des connaissances externes
        pour produire des réponses plus précises et actualisées.
        """,
        
        'architecture.txt': """
        L'architecture RAG modulaire se compose de plusieurs composants
        interchangeables: le loader pour charger les documents, le chunker
        pour les découper, l'embedder pour les vectoriser, le vector store
        pour les stocker, le retriever pour les rechercher, et le LLM pour
        générer les réponses finales.
        """,
        
        'benefits.txt': """
        Les principaux avantages du RAG incluent: la réduction des
        hallucinations des LLMs, l'accès à des connaissances actualisées,
        la traçabilité des sources, et la possibilité de travailler avec
        des domaines spécialisés sans fine-tuning complet du modèle.
        """
    }
    
    for filename, content in samples.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✅ {len(samples)} documents d'exemple créés dans {output_dir}")


if __name__ == "__main__":
    print(f"Architecture RAG v{__version__}")
    print("Ce module doit être importé, pas exécuté directement.")
    print("\nUsage:")
    print("  from src import quick_start")
    print("  pipeline = quick_start()")