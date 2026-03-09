# DAH Open RAG

**Système RAG open-source modulaire pour interroger vos documents en langage naturel.**

Construit par [Data Afrique Hub](https://github.com/abel2319) · Licence MIT

---

## Présentation

DAH Open RAG est un pipeline **Retrieval-Augmented Generation** clé-en-main.
Uploadez vos PDF, TXT ou Markdown, posez vos questions — le système cherche les passages pertinents dans vos documents et génère une réponse sourcée.

```
Question → Embedding → ChromaDB → Re-ranking → Mistral → Réponse + Sources
```

**Stack :**
| Couche | Technologie |
|---|---|
| API backend | FastAPI · Python 3.12 · uv |
| Embedding | `BAAI/bge-large-en-v1.5` (local, CPU) |
| Vector store | ChromaDB (persistant) |
| Re-ranking | `BAAI/bge-reranker-large` (local, CPU) |
| LLM | Mistral AI (`mistral-large-latest`) |
| Frontend | React 19 · Vite · Tailwind CSS |

---

## Démarrage rapide — Docker (recommandé)

### Prérequis
- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- Une clé API Mistral gratuite → [console.mistral.ai](https://console.mistral.ai)

### 1 · Configurer l'environnement

```bash
cp .env.example .env
# Éditez .env et renseignez votre MISTRAL_API_KEY
```

### 2 · Lancer

```bash
docker compose up --build
```

> Le premier démarrage télécharge les modèles d'embedding (~1,3 GB) — comptez 5-10 min.

- **Frontend** → [http://localhost](http://localhost)
- **API docs** → [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health** → [http://localhost:8000/health](http://localhost:8000/health)

### 3 · Utiliser

1. Ouvrez [http://localhost](http://localhost)
2. **Uploadez** un PDF/TXT/MD via le panneau gauche → cliquez **Indexer**
3. **Posez** vos questions dans le chat

---

## Démarrage manuel (développement)

### Backend

```bash
cd backend

# Installer les dépendances avec uv
uv sync

# Copier et remplir les variables d'environnement
cp ../.env.example .env
# → Renseignez MISTRAL_API_KEY dans .env

# Lancer le serveur (rechargement automatique)
uv run uvicorn src.api.main:app --reload
# API disponible sur http://localhost:8000
```

### Frontend

```bash
cd frontend

npm install

# Optionnel : pointer vers un backend distant
# echo "VITE_API_URL=http://mon-serveur:8000" > .env

npm run dev
# Interface disponible sur http://localhost:5173
```

---

## Structure du projet

```
dah-open-rag/
├── backend/
│   ├── src/
│   │   ├── api/            # Routes FastAPI (/query, /ingest, /health)
│   │   ├── core/           # Interfaces abstraites + orchestrateur
│   │   ├── llm/            # Adaptateurs LLM (Mistral, OpenAI, Ollama…)
│   │   ├── retrieval/      # Dense retriever, BM25, Hybrid, Reranker
│   │   ├── vectorstores/   # ChromaDB, FAISS
│   │   ├── Loaders/        # Chargement PDF, TXT, Markdown
│   │   ├── Chunkers/       # Découpage en chunks
│   │   ├── Embedders/      # SentenceTransformers
│   │   └── implementations/# Enregistrement des composants
│   ├── configs/
│   │   └── free.yaml       # Configuration active
│   ├── data/               # ChromaDB + uploads (ignoré par git)
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Interface principale
│   │   └── services/api.js # Client HTTP vers le backend
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## API

### `POST /query`
```json
{
  "question": "Quelles sont les données sur l'éducation au Sénégal ?",
  "chat_history": [],
  "top_k": 5,
  "rerank_top_k": 3,
  "llm_params": { "temperature": 0.7 }
}
```

**Réponse :**
```json
{
  "answer": "Selon les documents indexés…",
  "sources": [
    { "metadata": { "filename": "rapport_edu.pdf" }, "score": 0.92 }
  ]
}
```

### `POST /ingest`
Upload multipart de fichiers (PDF, TXT, MD).

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@mon_document.pdf"
```

### `GET /health`
```json
{ "status": "up", "stats": { "embedder": {…}, "vector_store": {…} } }
```

---

## Configuration

Tout se passe dans [`backend/configs/free.yaml`](backend/configs/free.yaml).
Changer un composant = changer une ligne, sans toucher au code.

```yaml
embedder:
  name: "sentence_transformers"
  params:
    model_name: "BAAI/bge-large-en-v1.5"  # ← changer ici

llm:
  name: "mistral"
  params:
    model: "mistral-large-latest"          # ← ou "mistral-small-latest"
    api_key: "${MISTRAL_API_KEY}"
```

---

## Contribuer

Les contributions sont les bienvenues !

1. Forkez le repo
2. Créez une branche : `git checkout -b feat/mon-ajout`
3. Committez vos changements
4. Ouvrez une **Pull Request**

Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les détails.

---

## Licence

[MIT](LICENSE) — Data Afrique Hub
