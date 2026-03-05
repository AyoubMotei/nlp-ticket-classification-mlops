# 📘 Product Requirements Document (PRD) - nlp-ticket-classification-mlops

## 1. Vision du Projet
L'objectif est d'industrialiser un pipeline de traitement de données (Batch) capable de transformer des emails de support client bruts en catégories classifiées, tout en garantissant la qualité du modèle et la stabilité de l'infrastructure via une stack MLOps moderne.

## 2. Objectifs Techniques & Métiers
- **Automatisation** : Passer d'une classification manuelle à une prédiction automatisée.
- **Sémantique** : Utiliser des embeddings (vecteurs) pour comprendre le sens des messages au-delà des mots.
- **Persistance** : Stocker les connaissances dans une base de données vectorielle (ChromaDB).
- **Observabilité** : Monitorer le "Drift" (dérive) des données et les performances CPU/RAM.

## 3. Spécifications Fonctionnelles (Étape par Étape)

### Étape 1 : Analyse & Préparation NLP
- **Entrée** : `data/dataset.csv`.
- **Logique** : 
    - Fusionner les colonnes `subject` et `body` dans `full_text`.
    - Nettoyage rigoureux : Minuscules, suppression ponctuation/caractères spéciaux.
    - **Gestion multilingue** : Suppression des stopwords Anglais ET Allemands.
- **Sortie** : `data/cleaned_dataset.csv`.

### Étape 2 : Vectorisation & Stockage Vectoriel
- **Modèle** : Hugging Face `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Logique** : 
    - Encoder `full_text` en vecteurs (embeddings).
    - Normaliser les vecteurs pour assurer la cohérence.
    - Créer/Mettre à jour une collection **ChromaDB** persistante dans `db/`.
- **Sortie** : Dossier `db/` contenant l'index vectoriel.

### Étape 3 : Classification supervisée
- **Modèle** : Scikit-learn (RandomForestClassifier ou SVM).
- **Logique** : 
    - Charger les vecteurs depuis ChromaDB.
    - Entraîner sur 80% des données, évaluer sur 20%.
    - Mesurer : F1-Score, Precision, Recall.
- **Sortie** : `models/classifier.pkl` (Artefact du modèle).

### Étape 4 : Monitoring ML (Evidently AI)
- **Logique** : 
    - Définir un jeu de référence (données d'entraînement).
    - Simuler des données actuelles pour détecter le **Data Drift** (changement de vocabulaire).
- **Sortie** : Un rapport interactif `reports/drift_report.html`.

### Étape 5 : Industrialisation & Orchestration
- **Docker** : Création d'une image capable d'exécuter tout le pipeline via un script maître.
- **Kubernetes** : Manifest `k8s/job.yaml` pour lancer le pipeline en mode Batch sur Minikube.
- **CI/CD** : Workflow GitHub Actions pour tester le code (Lint) et builder l'image Docker.

### Étape 6 : Monitoring Infrastructure
- **Prometheus** : Collecter les métriques via `cAdvisor` (Docker) et `Node Exporter` (Hôte).
- **Grafana** : Afficher l'usage CPU/RAM lors du traitement des embeddings.

## 4. Architecture des Données
| Champ | Type | Description |
| :--- | :--- | :--- |
| `subject` | string | Sujet de l'email |
| `body` | string | Corps du message |
| `full_text` | string | Fusion (Subject + Body) - Feature brute |
| `embeddings` | list(float) | Vecteur de dimension 384 - Feature calculée |
| `type` | string | Catégorie du ticket - Target (Label) |

## 5. Contraintes & Critères d'Acceptation
1. **Reproductibilité** : Le pipeline doit s'exécuter de bout en bout avec une seule commande Docker.
2. **Modularité** : Chaque script dans `scripts/` doit être testable individuellement.
3. **Persistance** : La base ChromaDB ne doit pas être perdue au redémarrage du container (usage de Volumes).
4. **Performance** : Le modèle doit atteindre un F1-Score > 0.80 sur le jeu de test.

## 6. Structure du Repo Cible
```text
nlp-ticket-classification-mlops/
├── data/           # CSV input/output
├── scripts/        # Python modules
├── db/             # ChromaDB files
├── models/         # Pickle files
├── reports/        # HTML Reports
├── monitoring/     # Docker-compose & Configs
├── k8s/            # Yaml manifests
├── Dockerfile      # Container config
└── requirements.txt