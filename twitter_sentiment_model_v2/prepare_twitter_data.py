import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def download_twitter_dataset():
    """
    Télécharge le dataset Twitter depuis Kaggle.
    Note: L'utilisateur doit avoir configuré ses credentials Kaggle.
    """
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            'kazanova/sentiment140',
            path='data',
            unzip=True
        )
        print("Dataset téléchargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        print("Veuillez télécharger manuellement le dataset depuis:")
        print("https://www.kaggle.com/datasets/kazanova/sentiment140")
        print("Et le placer dans le dossier 'data'")

def clean_tweet(text):
    """
    Nettoie un tweet en:
    - Supprimant les URLs
    - Supprimant les mentions @
    - Supprimant les hashtags #
    - Convertissant en minuscules
    - Supprimant la ponctuation
    - Supprimant les chiffres
    """
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression des mentions et hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Suppression des chiffres
    text = re.sub(r'\d+', '', text)
    
    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_tweet(text):
    """
    Prétraite un tweet en:
    - Tokenisant
    - Supprimant les stopwords
    - Lemmatisant
    """
    if not isinstance(text, str):
        return ""
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords et lemmatisation
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    
    return ' '.join(processed_tokens)

def prepare_dataset():
    """
    Prépare le dataset Twitter pour l'entraînement.
    """
    # Création du dossier data s'il n'existe pas
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Téléchargement du dataset
    download_twitter_dataset()
    
    # Chargement des données
    try:
        df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',
                        encoding='latin-1',
                        names=['target', 'id', 'date', 'flag', 'user', 'text'])
    except FileNotFoundError:
        print("Fichier non trouvé. Veuillez télécharger manuellement le dataset.")
        return None
    
    # Sélection des colonnes pertinentes
    df = df[['text', 'target']]
    
    # Conversion des targets (0 = négatif, 4 = positif)
    df['target'] = df['target'].map({0: 'negative', 4: 'positive'})
    
    # Nettoyage des tweets
    print("Nettoyage des tweets...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    # Prétraitement avancé
    print("Prétraitement avancé...")
    df['processed_text'] = df['cleaned_text'].apply(preprocess_tweet)
    
    # Suppression des lignes vides
    df = df[df['processed_text'].str.len() > 0]
    
    # Équilibrage des classes (prendre un nombre égal de tweets positifs et négatifs)
    min_class_size = min(df['target'].value_counts())
    df_balanced = pd.concat([
        df[df['target'] == 'positive'].sample(min_class_size),
        df[df['target'] == 'negative'].sample(min_class_size)
    ])
    
    # Sauvegarde du dataset prétraité
    df_balanced.to_csv('data/twitter_sentiment_processed.csv', index=False)
    print(f"Dataset prétraité sauvegardé avec {len(df_balanced)} tweets.")
    
    return df_balanced

if __name__ == "__main__":
    prepare_dataset() 