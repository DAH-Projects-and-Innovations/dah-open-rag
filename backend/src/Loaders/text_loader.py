import os
from typing import List
from pypdf import PdfReader
from langdetect import detect
from src.core.interfaces import IDocumentLoader
from src.core.models import Document

class UnifiedDocumentLoader(IDocumentLoader):
    def get_supported_formats(self) -> List[str]:
        return ["pdf", "md", "txt"]

    def load(self, source: str, **kwargs) -> List[Document]:
        all_documents = []
        # Supporter un dossier ou un fichier unique
        if os.path.isdir(source):
            files_to_process = [os.path.join(source, f) for f in os.listdir(source)]
        else:
            files_to_process = [source]

        for file_path in files_to_process:
            ext = file_path.split('.')[-1].lower()
            if ext not in self.get_supported_formats():
                continue

            content = ""
            try:
                if ext == 'pdf':
                    reader = PdfReader(file_path)
                    pages = [p.extract_text() for p in reader.pages]
                    content = "\n".join([t for t in pages if t])
                elif ext in ['md', 'txt']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
            except Exception as e:
                # Ignorer les fichiers qui posent problème mais continuer
                print(f"Warning: cannot read {file_path}: {e}")
                continue

            if not content or not content.strip():
                continue

            try:
                lang = detect(content[:1000])
            except Exception:
                lang = "unknown"

            all_documents.append(Document(
                content=content,
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "format": ext,
                    "lang": lang,
                }
            ))

        return all_documents