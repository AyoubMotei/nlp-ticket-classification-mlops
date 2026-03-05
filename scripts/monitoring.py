import pandas as pd
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer

"""
Ce script utilise Evidently AI pour monitorer le Data Drift et la Classification.
Note : En cas d'erreur d'importation sur certains environnements Windows, 
vérifier la compatibilité entre Pydantic v1 et Evidently.
"""

def main():
    print("📊 Test de présence d'Evidently...")
    try:
        # Importation dynamique (plus robuste)
        import evidently
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset
        from evidently.report import Report
        print(f"✅ Version détectée : {evidently.__version__}")
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return

    # 1. Chargement
    model_path = 'models/classifier.pkl'
    if not os.path.exists(model_path):
        print("❌ Modèle manquant. Lance train.py.")
        return

    df = pd.read_csv('data/cleaned_dataset.csv').dropna(subset=['cleaned_text'])
    clf = joblib.load(model_path)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 2. Simulation (Référence vs Actuel)
    # On prend peu de données pour que ce soit rapide
    ref_df = df.iloc[:300].copy()
    cur_df = df.iloc[300:600].copy()

    print("🔮 Calcul des prédictions...")
    ref_df['prediction'] = clf.predict(embed_model.encode(ref_df['cleaned_text'].tolist()))
    cur_df['prediction'] = clf.predict(embed_model.encode(cur_df['cleaned_text'].tolist()))
    
    ref_df['target'] = ref_df['type']
    cur_df['target'] = cur_df['type']

    # 3. Création du rapport
    print("📝 Analyse du Drift...")
    # On utilise une liste de métriques simple
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    report.run(
        reference_data=ref_df[['target', 'prediction']],
        current_data=cur_df[['target', 'prediction']]
    )

    # 4. Sauvegarde
    os.makedirs('reports', exist_ok=True)
    report.save_html('reports/monitoring_report.html')
    print("✨ SUCCÈS ! Ouvre le fichier : reports/monitoring_report.html")

if __name__ == "__main__":
    main()