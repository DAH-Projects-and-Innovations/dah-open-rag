"""
Système de gestion des prompts avec templates versionnés.
Support de variables, conditions et composition.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import re
import json


@dataclass
class PromptTemplate:
    """Template de prompt versionné."""
    name: str
    version: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def render(self, **kwargs) -> str:
        """
        Rend le template avec les variables fournies.
        
        Args:
            **kwargs: Variables à injecter dans le template
            
        Returns:
            Prompt rendu
        """
        # Vérifier les variables requises
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(
                f"Missing required variables: {missing_vars}"
            )
        
        # Remplacer les variables
        rendered = self.template
        for var, value in kwargs.items():
            # Support de différents formats: {var}, {{var}}, $var
            rendered = rendered.replace(f"{{{var}}}", str(value))
            rendered = rendered.replace(f"{{{{{var}}}}}", str(value))
            rendered = rendered.replace(f"${var}", str(value))
        
        return rendered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'name': self.name,
            'version': self.version,
            'template': self.template,
            'description': self.description,
            'variables': self.variables,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Crée depuis un dictionnaire."""
        return cls(**data)


class PromptManager:
    """
    Gestionnaire de prompts avec versioning.
    
    Permet de stocker, versionner et récupérer des templates de prompts.
    """
    
    def __init__(self):
        """Initialise le gestionnaire."""
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        # Structure: {name: {version: template}}
    
    def register_template(
        self,
        template: PromptTemplate,
        set_as_default: bool = True
    ) -> None:
        """
        Enregistre un template.
        
        Args:
            template: Template à enregistrer
            set_as_default: Si True, devient la version par défaut
        """
        if template.name not in self.templates:
            self.templates[template.name] = {}
        
        self.templates[template.name][template.version] = template
        
        # Marquer comme version par défaut
        if set_as_default:
            self.templates[template.name]['default'] = template
    
    def get_template(
        self,
        name: str,
        version: Optional[str] = None
    ) -> PromptTemplate:
        """
        Récupère un template.
        
        Args:
            name: Nom du template
            version: Version spécifique (None = version par défaut)
            
        Returns:
            Template
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        if version is None:
            version = 'default'
        
        if version not in self.templates[name]:
            available = [v for v in self.templates[name].keys() if v != 'default']
            raise ValueError(
                f"Version '{version}' not found for template '{name}'. "
                f"Available versions: {available}"
            )
        
        return self.templates[name][version]
    
    def render_template(
        self,
        name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Récupère et rend un template.
        
        Args:
            name: Nom du template
            version: Version spécifique
            **kwargs: Variables pour le rendu
            
        Returns:
            Prompt rendu
        """
        template = self.get_template(name, version)
        return template.render(**kwargs)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        Liste tous les templates disponibles.
        
        Returns:
            Liste de métadonnées des templates
        """
        result = []
        for name, versions in self.templates.items():
            for version, template in versions.items():
                if version == 'default':
                    continue
                result.append({
                    'name': name,
                    'version': version,
                    'description': template.description,
                    'variables': template.variables,
                    'is_default': template == versions.get('default')
                })
        return result
    
    def export_templates(self, filepath: str) -> None:
        """
        Exporte tous les templates vers un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        data = {}
        for name, versions in self.templates.items():
            data[name] = {}
            for version, template in versions.items():
                if version == 'default':
                    continue
                data[name][version] = template.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_templates(self, filepath: str) -> None:
        """
        Importe des templates depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier à importer
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, versions in data.items():
            for version, template_dict in versions.items():
                template = PromptTemplate.from_dict(template_dict)
                self.register_template(template, set_as_default=False)


# Templates RAG prédéfinis
RAG_SYSTEM_PROMPT = PromptTemplate(
    name="rag_system",
    version="1.0",
    template="""You are a helpful AI assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Answer ONLY using information from the context provided below
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Do not make up or invent information
4. Be concise and accurate
5. Cite the sources you use when answering

Context:
{context}""",
    description="System prompt pour RAG basique",
    variables=["context"]
)

RAG_USER_PROMPT = PromptTemplate(
    name="rag_user",
    version="1.0",
    template="""Question: {question}

Please provide a clear and accurate answer based on the context provided above.""",
    description="User prompt pour RAG basique",
    variables=["question"]
)

RAG_WITH_CITATIONS_SYSTEM = PromptTemplate(
    name="rag_citations_system",
    version="1.0",
    template="""You are a helpful AI assistant that provides accurate answers with proper citations.

IMPORTANT RULES:
1. Answer ONLY using information from the numbered sources below
2. ALWAYS cite your sources using [1], [2], etc. after each claim
3. If information is not in the sources, say "I don't have this information in the provided sources"
4. Never make up information
5. Be precise and factual

Sources:
{sources}""",
    description="System prompt pour RAG avec citations",
    variables=["sources"]
)

RAG_WITH_CITATIONS_USER = PromptTemplate(
    name="rag_citations_user",
    version="1.0",
    template="""Question: {question}

Please provide a detailed answer with citations [1], [2], etc. for each fact or claim you make.""",
    description="User prompt pour RAG avec citations",
    variables=["question"]
)

RAG_STRUCTURED_SYSTEM = PromptTemplate(
    name="rag_structured_system",
    version="1.0",
    template="""You are a helpful AI assistant that provides structured, well-organized answers.

IMPORTANT RULES:
1. Use ONLY information from the context below
2. Structure your answer clearly with sections if appropriate
3. Cite sources using [Source X] format
4. If uncertain, indicate your confidence level
5. Distinguish between facts and interpretations

Context:
{context}

Answer format:
- Start with a direct answer (1-2 sentences)
- Provide supporting details with citations
- End with a brief summary if the answer is long""",
    description="System prompt pour réponses structurées",
    variables=["context"]
)

# Prompt pour prévenir les hallucinations
RAG_SAFE_SYSTEM = PromptTemplate(
    name="rag_safe_system",
    version="1.0",
    template="""You are a careful AI assistant focused on accuracy and honesty.

CRITICAL SAFETY RULES:
1. NEVER provide information not explicitly stated in the context
2. If asked about something not in the context, respond: "This information is not available in my current context"
3. Do not use your general knowledge - ONLY use the provided context
4. If uncertain, say so explicitly
5. Distinguish clearly between what is stated in the context vs your interpretation

Context:
{context}

Remember: It's better to say "I don't know" than to provide incorrect information.""",
    description="System prompt avec sécurité renforcée contre les hallucinations",
    variables=["context"]
)

# Prompt multilingue
RAG_MULTILINGUAL_SYSTEM = PromptTemplate(
    name="rag_multilingual_system",
    version="1.0",
    template="""You are a multilingual AI assistant. Respond in the same language as the user's question.

Rules:
1. Answer based ONLY on the provided context
2. Respond in the same language as the question
3. Cite sources appropriately
4. If context is insufficient, say so in the user's language

Context:
{context}""",
    description="System prompt multilingue",
    variables=["context"]
)

# Prompt conversationnel
RAG_CONVERSATIONAL_SYSTEM = PromptTemplate(
    name="rag_conversational_system",
    version="1.0",
    template="""You are a friendly, conversational AI assistant.

Guidelines:
1. Be helpful and approachable in your tone
2. Use ONLY information from the context below
3. Provide clear, easy-to-understand explanations
4. Use examples when helpful
5. Acknowledge limitations honestly

Context:
{context}

Previous conversation:
{history}""",
    description="System prompt pour conversations naturelles",
    variables=["context", "history"]
)


def create_default_prompt_manager() -> PromptManager:
    """
    Crée un gestionnaire de prompts avec les templates par défaut.
    
    Returns:
        PromptManager avec templates prédéfinis
    """
    manager = PromptManager()
    
    # Enregistrer tous les templates par défaut
    default_templates = [
        RAG_SYSTEM_PROMPT,
        RAG_USER_PROMPT,
        RAG_WITH_CITATIONS_SYSTEM,
        RAG_WITH_CITATIONS_USER,
        RAG_STRUCTURED_SYSTEM,
        RAG_SAFE_SYSTEM,
        RAG_MULTILINGUAL_SYSTEM,
        RAG_CONVERSATIONAL_SYSTEM
    ]
    
    for template in default_templates:
        manager.register_template(template)
    
    return manager
