# Image de base optimisée pour Python
FROM python:3.11-slim

# Empêche Python d'écrire des fichiers .pyc et assure un affichage direct des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances système (nécessaires pour ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances
COPY requirements.txt .
# On force pydantic v1 avant le reste pour garantir que Evidently fonctionne
# RUN pip install --no-cache-dir "pydantic<2.0"
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du projet
COPY . .

# Par défaut, on lance le script d'entraînement
CMD ["python", "scripts/train.py"]