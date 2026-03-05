import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

def main():
    # Configuration des chemins
    input_file = 'data/cleaned_dataset.csv'
    db_path = 'db/' 
    
    if not os.path.exists(input_file):
        print(f"❌ Fichier {input_file} introuvable.")
        return
        
    # Chargement du dataset nettoyé
    df = pd.read_csv(input_file).dropna(subset=['cleaned_text'])
    
    # Chargement du modèle Hugging Face
    print("🧠 Chargement du modèle Hugging Face...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Initialisation de ChromaDB 
    print(f"📦 Initialisation de ChromaDB dans {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="support_tickets")

    documents = df['cleaned_text'].tolist()
    metadatas = [{"type": str(t)} for t in df['type']]
    ids = [str(i) for i in range(len(df))]

    print(f"🚀 Génération des embeddings pour {len(documents)} tickets...")
    embeddings = model.encode(documents, show_progress_bar=True)

    # Insertion par lots (Batching)
    batch_size = 4000
    print(f"📥 Stockage dans ChromaDB par lots de {batch_size}...")
    
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=documents[i:end]
        )
        print(f"✅ Lot {i//batch_size + 1} inséré...")

    print(f"✨ Étape 2 terminée ! {collection.count()} vecteurs indexés avec succès.")

if __name__ == "__main__":
    main()