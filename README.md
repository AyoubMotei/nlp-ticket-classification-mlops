# 🎫 nlp-ticket-classification-mlops

**Pipeline MLOps industriel : Classification sémantique de tickets support avec monitoring complet.**

## 1. 📋 Description du Workflow (Step-by-Step)

Le projet est divisé en 7 étapes critiques. Chaque script dans `scripts/` doit être autonome et pouvoir être appelé en mode "Batch".

### Étape 1 : Analyse & Preprocessing NLP (`scripts/preprocess.py`)
* **Action** : Charger `data/dataset.csv`.
* **Fusion** : Créer une colonne `full_text` = `subject` + `body`.
* **Nettoyage** : 
    - Minuscules, retrait ponctuation.
    - Tokenisation.
    - Suppression des stopwords (Anglais + Allemand via `nltk`).
* **Sortie** : Sauvegarder `data/cleaned_dataset.csv`.

### Étape 2 : Embeddings & Indexation Vectorielle (`scripts/embed_indexing.py`)
* **Modèle** : Utiliser `SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')`.
* **Action** : Transformer `cleaned_text` en vecteurs numériques (384 dimensions).
* **Base Vectorielle** : Initialiser **ChromaDB** en mode persistant dans le dossier `db/`.
* **Stockage** : Ajouter les vecteurs et les métadonnées (ID, label `type`) dans une collection nommée `ticket_embeddings`.

### Étape 3 : Entraînement du Classifieur (`scripts/train.py`)
* **Extraction** : Extraire les vecteurs et les labels de **ChromaDB**.
* **ML** : Split Train/Test (80/20). Entraîner un `RandomForestClassifier` de `scikit-learn`.
* **Évaluation** : Générer un rapport de classification (F1-score, Precision).
* **Persistance** : Sauvegarder le modèle dans `models/classifier.pkl`.

### Étape 4 : Monitoring de la Dérive (Drift) (`scripts/monitor.py`)
* **Outil** : `Evidently AI`.
* **Action** : Comparer le dataset d'entraînement (Reference) aux nouvelles données arrivantes (Current).
* **Rapport** : Générer un fichier `reports/drift_report.html` analysant le **Data Drift** et le **Target Drift**.

### Étape 5 : Conteneurisation (`Dockerfile`)
* **Image** : Base `python:3.10-slim`.
* **Instruction** : Installer les dépendances, copier les scripts et lancer le pipeline complet.
* **Optimisation** : Utiliser des volumes pour persister la base ChromaDB.

### Étape 6 : Orchestration Kubernetes (`k8s/job.yaml`)
* **Type** : Créer un **Kubernetes Job** pour exécuter le pipeline de façon isolée sur Minikube.
* **Config** : Mapper les fichiers via des ConfigMaps ou Volumes.

### Étape 7 : Monitoring Infrastructure (`monitoring/`)
* **Prometheus** : Configurer pour scraper les métriques de `cAdvisor` (consommation CPU/RAM du container).
* **Grafana** : Créer un dashboard pour visualiser la santé du pipeline pendant l'exécution.

## 📂 Structure Technique des Fichiers
```text
nlp-ticket-classification-mlops/
├── data/
│   └── dataset.csv             # Fichier source
├── scripts/
│   ├── preprocess.py           # Étape 1
│   ├── embed_indexing.py       # Étape 2
│   ├── train.py                # Étape 3
│   └── monitor.py              # Étape 4
├── models/
│   └── classifier.pkl          # Modèle entraîné
├── db/                         # Persistance ChromaDB
├── reports/                    # Rapports Evidently AI
├── monitoring/
│   ├── prometheus.yml          # Config Prometheus
│   └── docker-compose.yml      # Stack Grafana/Prometheus
├── k8s/
│   └── job.yaml                # Manifest Kubernetes
├── Dockerfile                  # Image Docker
├── requirements.txt            # Dépendances
└── README.md