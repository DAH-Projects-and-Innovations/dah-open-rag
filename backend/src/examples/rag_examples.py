"""
Exemples d'utilisation du moteur RAG.
Démontre les différents modes et configurations.
"""

import yaml
from typing import List

from src.rag.engine import RAGEngine, SimpleRAG, CitationRAG
from src.rag.models import RAGQuery, RAGConfig, Source
from src.llm.base_llm import create_llm, LLMConfig
from src.llm.prompt_manager import create_default_prompt_manager
from src.core.models import Document


def example_1_simple_rag():
    """
    Exemple 1: RAG simple sans citations.
    Pour réponses rapides et directes.
    """
    print("\n" + "="*60)
    print("EXEMPLE 1: RAG Simple (sans citations)")
    print("="*60)
    
    # 1. Préparer des documents d'exemple
    documents = [
        Document(
            id="doc1",
            content="Python est un langage de programmation de haut niveau créé par Guido van Rossum en 1991.",
            metadata={"source": "intro_python.txt", "score": 0.95}
        ),
        Document(
            id="doc2",
            content="Python est connu pour sa syntaxe claire et sa lisibilité, ce qui le rend idéal pour les débutants.",
            metadata={"source": "python_features.txt", "score": 0.89}
        ),
    ]
    
    print("\n📚 Documents disponibles:")
    for doc in documents:
        print(f"  - {doc.id}: {doc.content[:50]}...")
    
    # 2. Configuration
    print("\n⚙️ Configuration: Simple RAG")
    print("  - Citations: Non")
    print("  - Max tokens: 300")
    print("  - Temperature: 0.7")
    
    # 3. Exemple de réponse attendue
    print("\n❓ Question: Qu'est-ce que Python?")
    print("\n✅ Réponse attendue:")
    print("  Python est un langage de programmation de haut niveau créé en 1991.")
    print("  Il est réputé pour sa syntaxe claire et sa facilité d'apprentissage.")


def example_2_rag_with_citations():
    """
    Exemple 2: RAG avec citations complètes.
    Pour traçabilité et fiabilité.
    """
    print("\n" + "="*60)
    print("EXEMPLE 2: RAG avec Citations")
    print("="*60)
    
    # Documents
    documents = [
        Document(
            id="doc1",
            content="L'intelligence artificielle (IA) est définie comme la simulation de processus d'intelligence humaine par des machines.",
            metadata={"source": "AI_Definition.pdf", "score": 0.92}
        ),
        Document(
            id="doc2",
            content="Le machine learning est une branche de l'IA qui permet aux systèmes d'apprendre à partir de données sans être explicitement programmés.",
            metadata={"source": "ML_Intro.pdf", "score": 0.88}
        ),
        Document(
            id="doc3",
            content="Le deep learning utilise des réseaux de neurones profonds pour apprendre des représentations hiérarchiques des données.",
            metadata={"source": "DeepLearning_Guide.pdf", "score": 0.85}
        ),
    ]
    
    print("\n📚 Documents disponibles:")
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc.metadata['source']}")
        print(f"      {doc.content[:60]}...")
    
    print("\n⚙️ Configuration: RAG avec Citations")
    print("  - Citations: Oui [1], [2], [3]")
    print("  - Temperature: 0.5 (plus précis)")
    print("  - Seuil pertinence: 0.5")
    
    print("\n❓ Question: Qu'est-ce que le machine learning?")
    print("\n✅ Réponse attendue avec citations:")
    print("  Le machine learning est une branche de l'intelligence artificielle [1]")
    print("  qui permet aux systèmes d'apprendre à partir de données sans être")
    print("  explicitement programmés [2]. Cette approche se distingue du deep")
    print("  learning qui utilise des réseaux de neurones profonds [3].")
    print("\n  Sources:")
    print("  [1] AI_Definition.pdf")
    print("  [2] ML_Intro.pdf")
    print("  [3] DeepLearning_Guide.pdf")


def example_3_secure_rag():
    """
    Exemple 3: RAG sécurisé contre les hallucinations.
    Pour applications critiques.
    """
    print("\n" + "="*60)
    print("EXEMPLE 3: RAG Sécurisé (anti-hallucination)")
    print("="*60)
    
    # Documents limités
    documents = [
        Document(
            id="doc1",
            content="Le paracétamol est un médicament utilisé pour traiter la douleur et la fièvre.",
            metadata={"source": "medical_db.txt", "score": 0.88}
        ),
    ]
    
    print("\n📚 Documents disponibles:")
    print("  [1] medical_db.txt - Info sur paracétamol")
    
    print("\n⚙️ Configuration: RAG Sécurisé")
    print("  - Seuil pertinence: 0.7 (élevé)")
    print("  - Temperature: 0.3 (très déterministe)")
    print("  - Anti-hallucination: Activé")
    print("  - Require sources: Oui")
    
    print("\n❓ Question 1: Quelle est la posologie du paracétamol?")
    print("✅ Réponse sécurisée:")
    print("  Je n'ai pas assez d'informations dans ma base de connaissances")
    print("  pour répondre précisément à cette question. Les informations")
    print("  disponibles mentionnent que le paracétamol traite la douleur et")
    print("  la fièvre [1], mais la posologie n'est pas spécifiée.")
    print("  Veuillez consulter un professionnel de santé.")
    
    print("\n❓ Question 2: Comment fonctionnent les vaccins ARN?")
    print("✅ Réponse sécurisée:")
    print("  Je n'ai pas d'information sur ce sujet dans ma base de")
    print("  connaissances actuelle. Je ne peux pas répondre à cette question.")


def example_4_conversational_rag():
    """
    Exemple 4: RAG conversationnel avec historique.
    Pour dialogues naturels.
    """
    print("\n" + "="*60)
    print("EXEMPLE 4: RAG Conversationnel")
    print("="*60)
    
    print("\n💬 Historique de conversation:")
    conversation = [
        {"user": "Qu'est-ce que Python?", "assistant": "Python est un langage..."},
        {"user": "Qui l'a créé?", "assistant": "Guido van Rossum en 1991."},
        {"user": "Pourquoi est-il populaire?", "assistant": "..."},
    ]
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n  Tour {i}:")
        print(f"    👤 User: {turn['user']}")
        print(f"    🤖 Assistant: {turn['assistant']}")
    
    print("\n⚙️ Configuration: RAG Conversationnel")
    print("  - Historique: Utilisé pour contexte")
    print("  - Ton: Naturel et amical")
    print("  - Citations: Optionnelles selon le cas")


def example_5_structured_response():
    """
    Exemple 5: Réponses structurées en JSON.
    Pour intégrations et APIs.
    """
    print("\n" + "="*60)
    print("EXEMPLE 5: Réponses Structurées (JSON)")
    print("="*60)
    
    print("\n❓ Question: Quels sont les avantages de Python?")
    print("\n✅ Réponse structurée (JSON):")
    print("""{
  "answer": "Python offre plusieurs avantages majeurs",
  "key_points": [
    "Syntaxe claire et lisible",
    "Grande bibliothèque standard",
    "Large communauté de développeurs",
    "Polyvalent (web, data, IA, etc.)"
  ],
  "sources": ["doc1", "doc2", "doc3"],
  "citations": [
    {"point": "Syntaxe claire", "source": "[1]"},
    {"point": "Grande bibliothèque", "source": "[2]"}
  ],
  "confidence": "high",
  "related_topics": ["JavaScript", "Java", "Ruby"]
}""")


def example_6_domain_specific():
    """
    Exemple 6: RAG pour domaine spécifique.
    Avec prompt adapté.
    """
    print("\n" + "="*60)
    print("EXEMPLE 6: RAG Domaine Spécifique (Support Technique)")
    print("="*60)
    
    print("\n🎯 Domaine: Support Client - Produit SaaS")
    print("\n📚 Base de connaissances:")
    print("  - Articles KB-001 à KB-050")
    print("  - Guides de dépannage")
    print("  - FAQs produit")
    
    print("\n⚙️ Prompt personnalisé:")
    print('  "You are a friendly customer support agent for {company_name}."')
    print('  "Provide clear step-by-step solutions."')
    print('  "Reference KB articles with [KB-XXX]."')
    
    print("\n❓ Client: Mon application plante au démarrage")
    print("\n✅ Réponse Support:")
    print("  Bonjour! Je vais vous aider à résoudre ce problème.")
    print("")
    print("  Voici les étapes à suivre [KB-023]:")
    print("  1. Vérifiez que vous utilisez la dernière version")
    print("  2. Effacez le cache de l'application")
    print("  3. Redémarrez votre appareil")
    print("")
    print("  Si le problème persiste après ces étapes, je vais")
    print("  escalader votre cas à notre équipe technique.")


def example_7_multilingual():
    """
    Exemple 7: RAG multilingue.
    Réponses dans la langue de la question.
    """
    print("\n" + "="*60)
    print("EXEMPLE 7: RAG Multilingue")
    print("="*60)
    
    print("\n🌍 Support de langues: FR, EN, ES, DE")
    print("\n📚 Documents (multilingues):")
    print("  - EN: Python programming basics")
    print("  - FR: Introduction à Python")
    print("  - ES: Conceptos básicos de Python")
    
    print("\n❓ Question (EN): What is Python?")
    print("✅ Answer (EN):")
    print("  Python is a high-level programming language [1]...")
    
    print("\n❓ Question (FR): Qu'est-ce que Python?")
    print("✅ Réponse (FR):")
    print("  Python est un langage de programmation de haut niveau [1]...")
    
    print("\n❓ Pregunta (ES): ¿Qué es Python?")
    print("✅ Respuesta (ES):")
    print("  Python es un lenguaje de programación de alto nivel [1]...")


def example_8_confidence_levels():
    """
    Exemple 8: Niveaux de confiance dans les réponses.
    Transparence sur la certitude.
    """
    print("\n" + "="*60)
    print("EXEMPLE 8: Niveaux de Confiance")
    print("="*60)
    
    print("\n📊 Facteurs influençant la confiance:")
    print("  - Qualité des sources (score de pertinence)")
    print("  - Nombre de citations")
    print("  - Présence de phrases d'incertitude")
    print("  - Cohérence entre sources")
    
    print("\n✅ Confiance HAUTE:")
    print("  Question: Quand Python a-t-il été créé?")
    print("  Réponse: Python a été créé en 1991 [1][2]")
    print("  Confiance: HIGH (2 sources, score > 0.8)")
    
    print("\n⚠️ Confiance MOYENNE:")
    print("  Question: Python est-il plus rapide que Java?")
    print("  Réponse: Cela dépend du contexte [1]...")
    print("  Confiance: MEDIUM (comparaison nuancée)")
    
    print("\n❌ Confiance BASSE:")
    print("  Question: Quel est l'avenir de Python en 2030?")
    print("  Réponse: Je n'ai pas assez d'informations...")
    print("  Confiance: LOW/UNCERTAIN (spéculation)")


def example_9_load_from_config():
    """
    Exemple 9: Charger configuration depuis YAML.
    Configuration externalisée.
    """
    print("\n" + "="*60)
    print("EXEMPLE 9: Chargement depuis Configuration")
    print("="*60)
    
    configs = {
        "simple": "configs/rag/simple.yaml",
        "citations": "configs/rag/citations.yaml",
        "secure": "configs/rag/secure.yaml"
    }
    
    for name, path in configs.items():
        print(f"\n📄 {name.upper()}: {path}")
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"  ✓ Citations: {config.get('enable_citations', 'N/A')}")
            print(f"  ✓ Temperature: {config.get('default_temperature', 'N/A')}")
            print(f"  ✓ Top-k: {config.get('default_top_k', 'N/A')}")
            
            if 'description' in config:
                print(f"  ℹ️ {config['description'].strip()[:60]}...")
        except FileNotFoundError:
            print(f"  ⚠️ Fichier non trouvé")


def example_10_best_practices():
    """
    Exemple 10: Bonnes pratiques RAG.
    Recommandations et pièges à éviter.
    """
    print("\n" + "="*60)
    print("EXEMPLE 10: Bonnes Pratiques RAG")
    print("="*60)
    
    practices = [
        ("✅ DO", "Toujours citer les sources dans les réponses"),
        ("✅ DO", "Utiliser temperature basse (0.3-0.5) pour précision"),
        ("✅ DO", "Filtrer les sources par pertinence minimum"),
        ("✅ DO", "Indiquer explicitement les incertitudes"),
        ("✅ DO", "Tester avec des questions hors contexte"),
        ("❌ DON'T", "Ne jamais désactiver require_sources en prod"),
        ("❌ DON'T", "Ne pas utiliser temperature élevée (>0.8)"),
        ("❌ DON'T", "Ne pas ignorer les niveaux de confiance"),
        ("❌ DON'T", "Ne pas mélanger trop de domaines dans un RAG"),
    ]
    
    print("\n📋 Checklist:")
    for do_dont, practice in practices:
        print(f"  {do_dont}: {practice}")
    
    print("\n🎯 Quand utiliser quel mode?")
    print("  Simple RAG → FAQ, support basique")
    print("  Citation RAG → Documentation, recherche")
    print("  Secure RAG → Médical, légal, finance")
    print("  Conversational → Chat, assistant personnel")


if __name__ == "__main__":
    print("🔍 Exemples d'utilisation du moteur RAG")
    print("="*60)
    
    examples = [
        example_1_simple_rag,
        example_2_rag_with_citations,
        example_3_secure_rag,
        example_4_conversational_rag,
        example_5_structured_response,
        example_6_domain_specific,
        example_7_multilingual,
        example_8_confidence_levels,
        example_9_load_from_config,
        example_10_best_practices
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠️ Erreur: {e}")
    
    print("\n" + "="*60)
    print("✅ Tous les exemples terminés")
    print("="*60)
