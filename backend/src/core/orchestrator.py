import logging
from typing import Optional, Dict, Any, List
from .interfaces import (
    IDocumentLoader, IChunker, IEmbedder, IVectorStore,
    IRetriever, IReranker, IQueryRewriter, ILLM
)
from .models import Document, Chunk, Query, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrateur de pipeline RAG modulaire et configurable
    """
    
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        retriever: IRetriever,
        llm: ILLM,
        query_rewriter: Optional[IQueryRewriter] = None,
        reranker: Optional[IReranker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le pipeline RAG
        
        Args:
            embedder: Composant pour générer les embeddings
            vector_store: Composant pour stocker/rechercher les vecteurs
            retriever: Composant pour récupérer les documents
            llm: Composant pour générer les réponses
            query_rewriter: Composant optionnel pour réécrire les requêtes
            reranker: Composant optionnel pour réordonner les résultats
            config: Configuration du pipeline
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.config = config or {}
        
        logger.info(f"Pipeline RAG initialisé avec: "
                   f"embedder={type(embedder).__name__}, "
                   f"vector_store={type(vector_store).__name__}, "
                   f"retriever={type(retriever).__name__}, "
                   f"llm={type(llm).__name__}, "
                   f"query_rewriter={type(query_rewriter).__name__ if query_rewriter else None}, "
                   f"reranker={type(reranker).__name__ if reranker else None}")
    
    def ingest(
        self,
        loader: IDocumentLoader,
        chunker: IChunker,
        source: str,
        **kwargs
    ) -> int:
        """
        Ingère des documents dans le pipeline
        
        Args:
            loader: Loader pour charger les documents
            chunker: Chunker pour découper les documents
            source: Source des documents
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Nombre de chunks ingérés
        """
        logger.info(f"Début de l'ingestion depuis: {source}")
        
        # 1. Chargement des documents
        documents = loader.load(source, **kwargs.get('loader_params', {}))
        logger.info(f"{len(documents)} documents chargés")
        
        if not documents:
            logger.warning("Aucun document chargé")
            return 0
        
        # 2. Chunking
        chunks = chunker.chunk(documents, **kwargs.get('chunker_params', {}))
        logger.info(f"{len(chunks)} chunks créés")
        
        if not chunks:
            logger.warning("Aucun chunk créé")
            return 0
        
        # 3. Génération des embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        logger.info(f"Embeddings générés pour {len(chunks)} chunks")
        
        # 4. Stockage dans le vector store
        self.vector_store.add_chunks(chunks)
        logger.info(f"{len(chunks)} chunks stockés dans le vector store")
        
        return len(chunks)
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Exécute une requête sur le pipeline RAG
        
        Args:
            query_text: Texte de la requête
            top_k: Nombre de documents à récupérer
            rerank_top_k: Nombre de documents après reranking
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Réponse RAG avec sources
        """
        logger.info(f"Requête reçue: {query_text}")
        
        metadata = {
            'original_query': query_text,
            'steps': [],
            'top_k': top_k
        }
        
        # 1. Query Rewriting (optionnel)
        queries = [query_text]
        if self.query_rewriter:
            try:
                queries = self.query_rewriter.rewrite(query_text, **kwargs.get('rewriter_params', {}))
                logger.info(f"Requête réécrite en {len(queries)} variantes")
                metadata['steps'].append({'step': 'query_rewriting', 'num_queries': len(queries)})
            except Exception as e:
                logger.error(f"Erreur lors du query rewriting: {e}")
                queries = [query_text]
        
        # 2. Embedding et Retrieval
        all_documents = []
        for query in queries:
            try:
                query_obj = Query(text=query)
                query_obj.embedding = self.embedder.embed_query(query)
            except Exception as e:
                logger.error(f"Erreur lors de l'embedding pour '{query}': {e}")
            
            try:
                # 3. Retrieval
                docs = self.retriever.retrieve(query_obj, top_k=top_k, **kwargs.get('retriever_params', {}))
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Erreur lors de du retrieval pour '{query}': {e}")
            
        
        # Dédoublonnage
        seen_ids = set()
        unique_docs = []
        for doc in all_documents:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                unique_docs.append(doc)
        
        logger.info(f"{len(unique_docs)} documents uniques récupérés")
        metadata['steps'].append({'step': 'retrieval', 'num_docs': len(unique_docs)})

        # 4. Reranking (optionnel)
        if self.reranker and unique_docs:
            try:
                rerank_k = rerank_top_k or top_k
                #unique_docs = self.reranker.rerank(
                #    query_text, 
                #    unique_docs, 
                #    top_k=rerank_k,
                #    **kwargs.get('reranker_params', {})
                #)
                query_obj = Query(text=query_text)
                unique_docs = self.reranker.rerank(
                    query_obj,
                    unique_docs,
                    top_k=rerank_k,
                    **kwargs.get('reranker_params', {})
                )
                logger.info(f"Documents rerankés, top {len(unique_docs)} conservés")
                metadata['steps'].append({'step': 'reranking', 'num_docs': len(unique_docs)})
            except Exception as e:
                logger.error(f"Erreur lors du reranking: {e}")
        
        # 5. Génération de la réponse
        try:
            answer = self.llm.generate_with_context(
                query_text,
                unique_docs,
                **kwargs.get('llm_params', {})
            )
            logger.info("Réponse générée avec succès")
            metadata['steps'].append({'step': 'generation', 'answer_length': len(answer)})
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            answer = f"Erreur lors de la génération de la réponse: {str(e)}"
        
        # 6. Formatage avec citations (optionnel)
        if kwargs.get('include_citations', False) and unique_docs:
            answer = self._format_with_citations(answer, unique_docs)
            metadata['citations_added'] = True
        
        return RAGResponse(
            answer=answer,
            sources=unique_docs,
            metadata=metadata,
            query=query_text
        )
    
    def _format_with_citations(self, answer: str, sources: List[Document]) -> str:
        """
        Ajoute des citations numérotées à la fin de la réponse
        
        Args:
            answer: Réponse générée
            sources: Documents sources
            
        Returns:
            Réponse avec citations formatées
        """
        if not sources:
            return answer
        
        citations = "\n\n**Sources:**\n"
        for i, doc in enumerate(sources, 1):
            # Extrait les métadonnées pertinentes
            source_info = doc.metadata.get('source', 'Source inconnue')
            page = doc.metadata.get('page', '')
            
            citation = f"[{i}] {source_info}"
            if page:
                citation += f" (page {page})"
            # Récupérer le score de façon sûre (supporte Document ou Chunk-like)
            score_val = getattr(doc, 'score', None)
            if score_val is None:
                try:
                    score_val = float(doc.metadata.get('score')) if doc.metadata.get('score') is not None else None
                except Exception:
                    score_val = None

            if score_val is not None:
                try:
                    citation += f" (score: {score_val:.3f})"
                except Exception:
                    citation += f" (score: {score_val})"
            
            citations += f"{citation}\n"
        
        return answer + citations
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du pipeline"""
        try:
            stats = {
                'embedder': {
                    'type': type(self.embedder).__name__,
                    'dimension': self.embedder.get_dimension()
                },
                'vector_store': {
                    'type': type(self.vector_store).__name__,
                },
                'retriever': {
                    'type': type(self.retriever).__name__
                },
                'llm': {
                    'type': type(self.llm).__name__
                },
                'optional_components': {
                    'query_rewriter': type(self.query_rewriter).__name__ if self.query_rewriter else None,
                    'reranker': type(self.reranker).__name__ if self.reranker else None
                },
                'config': self.config
            }
            return stats
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {'error': str(e)}

