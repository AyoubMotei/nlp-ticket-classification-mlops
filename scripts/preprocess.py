import pandas as pd
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Téléchargement des ressources nécessaires
# On télécharge les 'stopwords' (mots fréquents sans valeur sémantique)
# et 'punkt' (le modèle pour découper les phrases en mots)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def clean_text(text):
    """Nettoyage NLP complet : minuscules, ponctuation, stopwords multilingues."""
    if not isinstance(text, str):
        return ""
    
    # 1. Mise en minuscules
    text = text.lower()
    
    # 2. Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenisation
    words = word_tokenize(text)
    
    # 4. Suppression des Stopwords (Anglais + Allemand)
    stop_words = set(stopwords.words('english') + stopwords.words('german'))
    cleaned_words = [w for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

def main():
    # Chemins des fichiers
    input_file = 'data/dataset.csv'
    output_file = 'data/cleaned_dataset.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Erreur : Le fichier {input_file} est introuvable.")
        return

    print(f"📖 Chargement des données depuis {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Fusion des champs (Gestion des NaN incluse)
    print("🔗 Fusion des champs 'subject' et 'body'...")
    df['full_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')

    # 2. Application du nettoyage NLP
    print("🧹 Nettoyage NLP en cours (cette étape peut prendre quelques instants)...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)

    # 3. Calcul de la longueur des emails (pour le futur monitoring)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    # 4. Sauvegarde du dataset nettoyé
    print(f"💾 Sauvegarde du dataset nettoyé dans {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("✅ Étape 1 terminée avec succès !")

if __name__ == "__main__":
    main()