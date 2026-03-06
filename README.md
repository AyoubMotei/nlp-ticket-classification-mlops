# 🎫 NLP Ticket Classification MLOps

> **Bienvenue dans ce projet d'industrialisation du Machine Learning !** 
> Ce dépôt est conçu comme un guide pédagogique pour comprendre comment transformer des données textuelles brutes (emails de support) en un système de classification automatisé, robuste et monitoré.

---

## 🌟 1. Vision & Contexte

Dans une entreprise, le support client reçoit des centaines d'emails par jour. Les trier manuellement est chronophage et source d'erreurs. 

**L'objectif de ce projet** est de construire une "usine" (Pipeline MLOps) qui :
1. **Comprend** le sens des messages (NLP).
2. **Apprend** à les catégoriser (Machine Learning).
3. **S'auto-surveille** pour détecter si le langage des clients change (Monitoring).
4. **S'exécute partout** de manière identique (Docker & Kubernetes).

---

## 🏗️ 2. Architecture du Projet

Le projet suit une structure modulaire où chaque étape est isolée et reproductible.

```text
nlp-ticket-classification-mlops/
├── data/           # 📦 Données brutes et nettoyées (CSV)
├── scripts/        # ⚙️ Le "moteur" : scripts Python par étape
├── db/             # 🧠 Mémoire sémantique (Base de données ChromaDB)
├── models/         # 🤖 Intelligence stockée (Modèle entraîné .pkl)
├── reports/        # 📊 Analyses et rapports de monitoring
├── monitoring/     # 🖥️ Infrastructure d'observabilité (Prometheus/Grafana)
├── k8s/            # ⛵ Orchestration Kubernetes
├── Dockerfile      # 🐳 Recette de cuisine pour le container
└── requirements.txt # 📋 Liste des ingrédients (bibliothèques)
```

---

## 🔍 3. Analyse détaillée des composants (Step-by-Step)

### Étape 1 : Préparation des données (`scripts/preprocess.py`)
Le texte brut est "sale" (majuscules, ponctuation, mots inutiles). Ce script :
- **Fusionne** le sujet et le corps de l'email pour avoir tout le contexte.
- **Normalise** : Passage en minuscules et retrait de la ponctuation.
- **Filtre** : Supprime les "Stopwords" (mots comme *le, la, und, the*) en Anglais et Allemand.
- **Résultat** : Un fichier `data/cleaned_dataset.csv` prêt pour l'IA.

### Étape 2 : Mémoire Vectorielle (`scripts/embed_indexing.py`)
Les ordinateurs ne comprennent pas les mots, mais les nombres.
- **Embeddings** : On utilise le modèle Hugging Face `paraphrase-multilingual-MiniLM-L12-v2` pour transformer chaque phrase en un vecteur de 384 nombres. Deux phrases ayant un sens proche auront des vecteurs proches.
- **ChromaDB** : Ces vecteurs sont stockés dans une base de données vectorielle persistante. C'est la "mémoire sémantique" du projet.

### Étape 3 : Apprentissage du Cerveau (`scripts/train.py`)
- **Extraction** : On récupère les vecteurs et leurs catégories depuis ChromaDB.
- **Entraînement** : On utilise une `LogisticRegression` (scikit-learn) pour apprendre à associer un vecteur à une catégorie (ex: "Facturation", "Technique").
- **Évaluation** : Le script affiche un rapport de performance (F1-Score).
- **Sauvegarde** : L'intelligence est figée dans `models/classifier.pkl`.

### Étape 4 : Surveillance (`scripts/monitoring.py`)
Le monde change, les mots aussi.
- **Evidently AI** : Compare les données passées avec les nouvelles.
- **Data Drift** : Détecte si le vocabulaire des clients dérive, ce qui pourrait rendre le modèle obsolète.
- **Rapport** : Génère un dashboard visuel dans `reports/monitoring_report.html`.

---

## 🐳 4. Industrialisation & Infrastructure

### Conteneurisation (`Dockerfile`)
Le `Dockerfile` définit l'environnement exact de calcul. Il installe le système, les dépendances Python, et prépare le projet pour qu'il s'exécute de la même manière sur votre PC que sur un serveur cloud.

### Orchestration (`k8s/`)
Pour automatiser l'exécution à grande échelle :
- **Jobs** : Lancement ponctuel du pipeline de ré-entraînement.
- **Persistence** : Utilisation de volumes pour que les modèles et la base de données ne disparaissent pas à l'arrêt du container.

### Monitoring Infrastructure (`monitoring/`)
- **Prometheus** : Récupère les métriques de performance (CPU, RAM).
- **Grafana** : Affiche ces données sur des graphiques en temps réel.

---

## 🚀 5. Guide d'Utilisation rapide

### Pré-requis
Avoir **Docker** installé sur votre machine.

### Lancement du Pipeline complet
```bash
# 1. Construire l'image Docker
docker build -t nlp-mlops .

# 2. Lancer le pipeline complet
docker run -v $(pwd)/db:/app/db -v $(pwd)/models:/app/models nlp-mlops
```

### Lancement du Monitoring Infra
```bash
cd monitoring
docker-compose up -d
```
Puis accédez à Grafana sur `http://localhost:3000`.

---

## 🎓 6. Concepts Clés pour apprendre
- **Embeddings** : Représentation mathématique du sens d'un texte.
- **Vector Database** : Base de données spécialisée pour la recherche par similarité.
- **MLOps** : Fusion du Machine Learning et du DevOps pour garantir des modèles fiables en production.
