from typing import List
from src.core.interfaces import IChunker
from src.core.models import Document, Chunk

class ConfigurableChunker(IChunker):
    def chunk(self, documents: List[Document], **kwargs) -> List[Chunk]:
        size = kwargs.get('chunk_size', 500)
        overlap = kwargs.get('chunk_overlap', 50)
        all_chunks = []

        for doc in documents:
            text = doc.content
            start = 0
            while start < len(text):
                end = start + size
                chunk_text = text[start:end]
                
                # Pour crée le chunk en liant le doc_id du parent
                new_chunk = Chunk(
                    content=chunk_text,
                    doc_id=doc.doc_id,
                    metadata={**doc.metadata, "chunk_size": len(chunk_text)}
                )
                all_chunks.append(new_chunk)
                
                # pour le Calcul du prochain départ avec overlap
                start += (size - overlap)
                
        return all_chunks