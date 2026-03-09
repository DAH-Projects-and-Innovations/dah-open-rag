# DAH Open RAG

**Système RAG open-source modulaire pour interroger vos documents en langage naturel.**

Construit par [Data Afrique Hub](https://github.com/abel2319) · Licence MIT

---

## Présentation

DAH Open RAG est un pipeline **Retrieval-Augmented Generation** clé-en-main.
Uploadez vos PDF, TXT ou Markdown, posez vos questions — le système trouve les passages pertinents et génère une réponse sourcée.

```
Question → Embedding → ChromaDB → Re-ranking → LLM → Réponse + Sources
```

Deux configurations sont disponibles :

| | Config `free` | Config `hybrid` |
|---|---|---|
| **LLM** | Ollama (local) | Mistral ou Gemini (API cloud) |
| **Clé API** | Aucune | Oui (gratuite) |
| **Embedding** | `bge-small-en-v1.5` (local) | `bge-large-en-v1.5` (local) |
| **Reranking** | `ms-marco-MiniLM` (local) | `bge-reranker-large` (local) |
| **Coût** | 0 € | ~0 € avec quota gratuit |
| **Qualité réponse** | Bonne | Excellente |
| **RAM requise** | ~6 GB | ~4 GB |

---

## Prérequis

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)

---

## Mode HYBRID — Mistral ou Gemini (recommandé)

### 1. Obtenir une clé API gratuite

- **Mistral** → [console.mistral.ai](https://console.mistral.ai)
- **Gemini** → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 2. Configurer

```bash
cp .env.example .env
```

Éditez `.env` :

```env
RAG_CONFIG=hybrid

# Avec Mistral :
MISTRAL_API_KEY=votre_clé_mistral

# OU avec Gemini (voir instructions ci-dessous) :
# GEMINI_API_KEY=votre_clé_gemini
```

**Utiliser Gemini :** ouvrez `backend/configs/hybrid.yaml`, commentez le bloc `llm: mistral` et décommentez le bloc `llm: gemini`.

### 3. Lancer

```bash
docker compose up --build
```

> Premier démarrage : téléchargement des modèles d'embedding (~1.3 GB), ~5 min.

- **Interface** → http://localhost
- **API docs** → http://localhost:8000/docs

---

## Mode FREE — Ollama (100 % local, aucune clé API)

### 1. Configurer

```bash
cp .env.example .env
```

Éditez `.env` :

```env
RAG_CONFIG=free
```

### 2. Lancer (backend + frontend + Ollama)

```bash
docker compose -f docker-compose.yml -f docker-compose.free.yml up --build
```

> Premier démarrage : téléchargement d'Ollama + modèle `llama3.1:8b` (~4.7 GB), ~10-20 min.
> Les démarrages suivants sont instantanés (modèles mis en cache).

- **Interface** → http://localhost
- **API docs** → http://localhost:8000/docs

### Variante : Ollama déjà installé localement

```bash
# Télécharger le modèle une seule fois
ollama pull llama3.1:8b

# Lancer uniquement backend + frontend
RAG_CONFIG=free docker compose up --build
```

---

## Développement sans Docker

### Backend

```bash
cd backend
uv sync
cp ../.env.example .env   # puis remplissez .env
uv run uvicorn src.api.main:app --reload
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Utilisation

1. Ouvrez l'interface dans votre navigateur
2. **Uploadez** un fichier PDF, TXT ou Markdown (panneau gauche)
3. Cliquez **Indexer dans la base**
4. **Posez vos questions** — la réponse affiche les sources utilisées

---

## Structure du projet

```
dah-open-rag/
├── docker-compose.yml          ← Mode HYBRID
├── docker-compose.free.yml     ← Override mode FREE (+ Ollama)
├── .env.example
│
├── backend/
│   ├── configs/
│   │   ├── free.yaml           ← Ollama local
│   │   └── hybrid.yaml         ← Mistral ou Gemini
│   └── src/
│       ├── api/                ← Routes FastAPI
│       ├── core/               ← Interfaces + orchestrateur
│       ├── llm/                ← Adaptateurs (Ollama, Mistral, Gemini…)
│       ├── retrieval/          ← Dense retriever, Reranker
│       ├── vectorstores/       ← ChromaDB
│       ├── Loaders/            ← PDF, TXT, Markdown
│       └── Chunkers/
│
└── frontend/
    └── src/
        ├── App.jsx
        └── services/api.js
```

---

## Changer de modèle

Sans toucher au code, modifiez simplement le YAML :

```yaml
# free.yaml — changer le modèle Ollama
llm:
  params:
    model: "mistral:7b"          # ou gemma2:9b, phi3, qwen2.5, etc.

# hybrid.yaml — changer le modèle Mistral
llm:
  params:
    model: "mistral-large-latest"
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

### `POST /ingest`
```bash
curl -X POST http://localhost:8000/ingest -F "files=@rapport.pdf"
```

### `GET /health`
```json
{ "status": "up", "stats": { ... } }
```

---

## Contribuer

1. Forkez le repo
2. `git checkout -b feat/mon-ajout`
3. Ouvrez une **Pull Request**

---

## Licence

[MIT](LICENSE) — Data Afrique Hub
