import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Type, Optional
import logging

from .orchestrator import RAGPipeline
from .interfaces import (
    IDocumentLoader, IChunker, IEmbedder, IVectorStore,
    IRetriever, IReranker, IQueryRewriter, ILLM
)

from src.llm.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RAGPipelineFactory:
    """Factory pour créer des pipelines RAG à partir de configurations"""
    
    # Registre des implémentations disponibles
    _registry: Dict[str, Dict[str, Type]] = {
        'embedders': {},
        'vector_stores': {},
        'retrievers': {},
        'rerankers': {},
        'query_rewriters': {},
        'llms': {},
        'loaders': {},
        'chunkers': {},
        'prompt_managers': {}
    }
    
    @classmethod
    def register_component(cls, component_type: str, name: str, component_class: Type):
        """
        Enregistre une implémentation de composant
        
        Args:
            component_type: Type de composant (embedders, vector_stores, etc.)
            name: Nom de l'implémentation
            component_class: Classe à enregistrer
        """
        if component_type not in cls._registry:
            raise ValueError(f"Type de composant inconnu: {component_type}")
        
        cls._registry[component_type][name] = component_class
        logger.info(f"Composant enregistré: {component_type}/{name}")
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Charge une configuration depuis un fichier YAML ou JSON
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            Dictionnaire de configuration
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Format non supporté: {path.suffix}")
        
        # Remplacement des variables d'environnement
        config = cls._replace_env_vars(config)
        
        logger.info(f"Configuration chargée depuis: {config_path}")
        return config
    
    @classmethod
    def _replace_env_vars(cls, config: Any) -> Any:
        """Remplace les variables d'environnement dans la configuration"""
        if isinstance(config, dict):
            return {k: cls._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> RAGPipeline:
        """
        Crée un pipeline RAG à partir d'une configuration
        
        Args:
            config: Dictionnaire de configuration
            
        Returns:
            Pipeline RAG configuré
        """
        logger.info("Création du pipeline depuis la configuration")
        
        # Création des composants obligatoires
        embedder = cls._create_component('embedders', config['embedder'])
        vector_store = cls._create_component('vector_stores', config['vector_store'])
        config['retriever']['params']['vector_store'] = vector_store
        config['retriever']['params']['embedder'] = embedder
        retriever = cls._create_component('retrievers', config['retriever'])

        # 1. Créer le PromptManager d'abord
        prompt_manager = cls._create_component('prompt_managers', config['prompt_managers'])
        
        # 2. Passer le prompt_manager aux params du LLM avant création
        llm_config = config['llm']
        llm_config['params']['prompt_manager'] = prompt_manager
        llm_config['params']['provider'] = llm_config['name'] # ex: 'ollama'
        llm = cls._create_component('llms', llm_config)
        
        # Création des composants optionnels
        query_rewriter = None
        if 'query_rewriter' in config:
            query_rewriter = cls._create_component('query_rewriters', config['query_rewriter'])
        
        reranker = None
        if 'reranker' in config:
            reranker = cls._create_component('rerankers', config['reranker'])
        
        # Création du pipeline
        pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            llm=llm,
            query_rewriter=query_rewriter,
            reranker=reranker,
            config=config.get('pipeline_config', {})
        )
        
        logger.info("Pipeline créé avec succès")
        return pipeline
    
    @classmethod
    def _create_component(cls, component_type: str, component_config: Dict[str, Any]):
        """
        Crée un composant à partir de sa configuration
        
        Args:
            component_type: Type de composant
            component_config: Configuration du composant
            
        Returns:
            Instance du composant
        """
        
        name = component_config['name']
        params = component_config.get('params', {})
        
        if name not in cls._registry[component_type]:
            available = ', '.join(cls._registry[component_type].keys())
            raise ValueError(
                f"Composant non enregistré: {component_type}/{name}. "
                f"Disponibles: {available}"
            )
        
        component_class = cls._registry[component_type][name]
        logger.debug(f"Création du composant: {component_type}/{name}")
        
        try:
            return component_class(**params)
        except Exception as e:
            logger.error(f"Erreur lors de la création de {component_type}/{name}: {e}")
            raise
    
    @classmethod
    def list_components(cls, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        Liste les composants enregistrés
        
        Args:
            component_type: Type spécifique ou None pour tous
            
        Returns:
            Dictionnaire des composants disponibles
        """
        if component_type:
            if component_type not in cls._registry:
                raise ValueError(f"Type inconnu: {component_type}")
            return {component_type: list(cls._registry[component_type].keys())}
        
        return {k: list(v.keys()) for k, v in cls._registry.items()}
