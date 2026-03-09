FROM python:3.12-slim

WORKDIR /app

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copier les fichiers de dépendances en premier (cache Docker)
COPY pyproject.toml uv.lock ./

# Installer les dépendances sans le projet lui-même
RUN uv sync --frozen --no-install-project

# Copier le code source
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py ./

# Créer les dossiers de données persistantes
RUN mkdir -p data/chroma_db data/uploads

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
