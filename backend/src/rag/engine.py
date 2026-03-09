"""
Moteur RAG avec génération contrôlée et citations.
Support de différents modes: basique, avec citations, structuré.
"""

import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.interfaces import IRetriever
from ..core.models import Document, Query
from ..llm.base_llm import BaseLLM, LLMMessage
from ..llm.prompt_manager import PromptManager
from .models import (
    RAGResponse,
    RAGQuery,
    RAGConfig,
    Source,
    Citation,
    ConfidenceLevel
)


class RAGEngine:
    """
    Moteur RAG avec génération contrôlée.
    
    Fournit des réponses fiables avec citations et traçabilité.
    """
    
    def __init__(
        self,
        retriever: IRetriever,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialise le moteur RAG.
        
        Args:
            retriever: Système de retrieval
            llm: Modèle de langage
            prompt_manager: Gestionnaire de prompts
            config: Configuration RAG
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.config = config or RAGConfig()
    
    def query(
        self,
        query: RAGQuery,
        **kwargs
    ) -> RAGResponse:
        """
        Traite une requête RAG complète.
        
        Args:
            query: Requête utilisateur
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse RAG avec sources et citations
        """
        start_time = time.time()
        
        # 1. Retrieval
        retrieved_docs = self._retrieve_documents(query)
        
        # 2. Vérifier qu'on a des sources
        if not retrieved_docs and self.config.require_sources:
            return self._create_no_source_response(query)
        
        # 3. Préparer le contexte
        sources = self._prepare_sources(retrieved_docs)
        context = self._format_context(sources, query.include_citations)
        
        # 4. Générer la réponse
        llm_response = self._generate_answer(
            question=query.question,
            context=context,
            include_citations=query.include_citations,
            temperature=query.temperature,
            max_tokens=query.max_answer_length
        )
        
        # 5. Extraire les citations si demandées
        citations = []
        if query.include_citations:
            citations = self._extract_citations(llm_response.content, sources)
        
        # 6. Évaluer la confiance
        confidence = self._evaluate_confidence(
            llm_response.content,
            sources,
            citations
        )
        
        # 7. Détecter les réponses hors contexte
        based_on_context = self._verify_context_usage(
            llm_response.content,
            sources
        )
        
        # 8. Construire la réponse
        generation_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            citations=citations,
            confidence=confidence,
            based_on_context=based_on_context,
            model_used=llm_response.model,
            tokens_used=llm_response.usage['total_tokens'],
            generation_time_ms=generation_time,
            metadata={
                'retrieval_count': len(retrieved_docs),
                'finish_reason': llm_response.finish_reason
            }
        )
    
    def _retrieve_documents(self, query: RAGQuery) -> List[Document]:
        """
        Récupère les documents pertinents.
        
        Args:
            query: Requête RAG
            
        Returns:
            Documents récupérés
        """
        retrieval_query = Query(
            text=query.question,
            filters=query.retrieval_filters
        )
        
        documents = self.retriever.retrieve(
            query=retrieval_query,
            top_k=query.top_k
        )
        
        # Filtrer par score minimum si configuré
        if self.config.min_relevance_score > 0:
            documents = [
                doc for doc in documents
                if doc.metadata.get('score', 0) >= self.config.min_relevance_score
            ]
        
        return documents
    
    def _prepare_sources(self, documents: List[Document]) -> List[Source]:
        """
        Convertit les documents en sources avec numéros de citation.
        
        Args:
            documents: Documents récupérés
            
        Returns:
            Sources formatées
        """
        sources = []
        for i, doc in enumerate(documents, start=1):
            source = Source(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                relevance_score=doc.metadata.get('score', 0),
                citation_id=i
            )
            sources.append(source)
        
        return sources
    
    def _format_context(
        self,
        sources: List[Source],
        include_citations: bool
    ) -> str:
        """
        Formate le contexte pour le LLM.
        
        Args:
            sources: Sources à inclure
            include_citations: Si True, numéroter les sources
            
        Returns:
            Contexte formaté
        """
        if include_citations:
            # Format avec numéros de citation
            context_parts = []
            for source in sources:
                header = f"[{source.citation_id}]"
                if 'title' in source.metadata:
                    header += f" {source.metadata['title']}"
                context_parts.append(f"{header}\n{source.content}")
            
            return "\n\n".join(context_parts)
        else:
            # Format simple sans citations
            return "\n\n".join(s.content for s in sources)
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        include_citations: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Any:
        """
        Génère la réponse via le LLM.
        
        Args:
            question: Question de l'utilisateur
            context: Contexte (sources)
            include_citations: Si True, demander des citations
            temperature: Température de génération
            max_tokens: Nombre max de tokens
            
        Returns:
            Réponse du LLM
        """
        # Choisir les prompts appropriés
        if include_citations:
            system_template = self.config.system_prompt_template
            user_template = self.config.user_prompt_template
        else:
            system_template = "rag_system"
            user_template = "rag_user"
        
        # Rendre les prompts
        if include_citations:
            system_prompt = self.prompt_manager.render_template(
                system_template,
                sources=context
            )
        else:
            system_prompt = self.prompt_manager.render_template(
                system_template,
                context=context
            )
        
        user_prompt = self.prompt_manager.render_template(
            user_template,
            question=question
        )
        
        # Créer les messages
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        # Générer
        return self.llm.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _extract_citations(
        self,
        answer: str,
        sources: List[Source]
    ) -> List[Citation]:
        """
        Extrait les citations de la réponse.
        
        Args:
            answer: Réponse générée
            sources: Sources disponibles
            
        Returns:
            Liste de citations trouvées
        """
        citations = []
        
        # Trouver toutes les citations [1], [2], etc.
        pattern = r'\[(\d+)\]'
        matches = re.finditer(pattern, answer)
        
        for match in matches:
            citation_num = int(match.group(1))
            
            # Trouver la source correspondante
            source = next(
                (s for s in sources if s.citation_id == citation_num),
                None
            )
            
            if source:
                # Extraire un snippet autour de la citation
                start = max(0, match.start() - 50)
                end = min(len(answer), match.end() + 50)
                snippet = answer[start:end]
                
                citation = Citation(
                    source_id=source.id,
                    citation_number=citation_num,
                    text_snippet=snippet,
                    position_in_answer=match.start()
                )
                citations.append(citation)
        
        return citations
    
    def _evaluate_confidence(
        self,
        answer: str,
        sources: List[Source],
        citations: List[Citation]
    ) -> ConfidenceLevel:
        """
        Évalue le niveau de confiance dans la réponse.
        
        Args:
            answer: Réponse générée
            sources: Sources utilisées
            citations: Citations trouvées
            
        Returns:
            Niveau de confiance
        """
        # Indicateurs de faible confiance
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i don't have",
            "information is not available",
            "cannot answer",
            "insufficient information",
            "je ne sais pas",
            "je ne suis pas sûr"
        ]
        
        answer_lower = answer.lower()
        
        # Vérifier les phrases d'incertitude
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return ConfidenceLevel.UNCERTAIN
        
        # Pas de sources
        if not sources:
            return ConfidenceLevel.LOW
        
        # Avec citations
        if citations:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
            citation_ratio = len(citations) / len(sources)
            
            if avg_relevance > 0.8 and citation_ratio > 0.5:
                return ConfidenceLevel.HIGH
            elif avg_relevance > 0.6:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        
        # Sans citations mais avec sources
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        if avg_relevance > 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _verify_context_usage(
        self,
        answer: str,
        sources: List[Source]
    ) -> bool:
        """
        Vérifie que la réponse est basée sur le contexte.
        
        Args:
            answer: Réponse générée
            sources: Sources fournies
            
        Returns:
            True si basé sur le contexte
        """
        if not self.config.prevent_hallucinations:
            return True
        
        # Phrases indiquant une réponse hors contexte
        out_of_context_phrases = [
            "based on my knowledge",
            "in general",
            "typically",
            "commonly",
            "generally speaking",
            "d'après mes connaissances",
            "en général"
        ]
        
        answer_lower = answer.lower()
        
        # Si on trouve ces phrases, c'est suspect
        if any(phrase in answer_lower for phrase in out_of_context_phrases):
            return False
        
        # TODO: Analyse plus sophistiquée (overlap de tokens, etc.)
        
        return True
    
    def _create_no_source_response(self, query: RAGQuery) -> RAGResponse:
        """
        Crée une réponse quand aucune source n'est trouvée.
        
        Args:
            query: Requête originale
            
        Returns:
            Réponse indiquant l'absence de sources
        """
        return RAGResponse(
            answer="I don't have enough information in my knowledge base to answer this question accurately. Please try rephrasing your question or provide more context.",
            sources=[],
            citations=[],
            confidence=ConfidenceLevel.UNCERTAIN,
            based_on_context=False,
            metadata={'reason': 'no_sources_found'}
        )
    
    def update_config(self, config: RAGConfig) -> None:
        """
        Met à jour la configuration du moteur.
        
        Args:
            config: Nouvelle configuration
        """
        self.config = config


class SimpleRAG(RAGEngine):
    """
    Version simplifiée du RAG sans citations.
    
    Pour les cas d'usage où la traçabilité n'est pas critique.
    """
    
    def query(self, query: RAGQuery, **kwargs) -> RAGResponse:
        """Traite une requête en mode simple (sans citations)."""
        # Forcer la désactivation des citations
        query.include_citations = False
        return super().query(query, **kwargs)


class CitationRAG(RAGEngine):
    """
    Version du RAG optimisée pour les citations.
    
    Force toujours l'inclusion de citations et vérifie leur présence.
    """
    
    def query(self, query: RAGQuery, **kwargs) -> RAGResponse:
        """Traite une requête en mode citations."""
        # Forcer l'activation des citations
        query.include_citations = True
        
        response = super().query(query, **kwargs)
        
        # Vérifier la présence de citations
        if response.answer and not response.citations:
            # Ajouter un avertissement
            response.metadata['warning'] = 'No citations found in answer'
            response.confidence = ConfidenceLevel.LOW
        
        return response
