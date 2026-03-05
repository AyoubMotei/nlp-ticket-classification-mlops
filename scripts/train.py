import chromadb
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    db_path = 'db/'
    model_path = 'models/classifier.pkl'
    
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Dossier 'models/' créé.")

    print("🔌 Connexion à ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="support_tickets")
        
        # Récupération des données
        results = collection.get(include=['embeddings', 'metadatas'])
        
        X = results['embeddings']
        y = [m['type'] for m in results['metadatas']]
        
        # --- CORRECTION : Utilisation de len() pour éviter l'ambiguïté ---
        if len(X) == 0:
            print("❌ Erreur : La base de données est vide. Relance embed_indexing.py.")
            return
            
    except Exception as e:
        print(f"❌ Erreur lors de l'accès à ChromaDB : {e}")
        return

    # conversion en array numpy pour plus de stabilité
    X = np.array(X)
    y = np.array(y)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🏗️ Entraînement du modèle sur {len(X_train)} tickets...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\n📊 ÉVALUATION DU MODÈLE :")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"💾 Sauvegarde du modèle dans {model_path}...")
    joblib.dump(clf, model_path)
    
    print("\n✅ Étape 3 terminée avec succès !")

if __name__ == "__main__":
    main()
    