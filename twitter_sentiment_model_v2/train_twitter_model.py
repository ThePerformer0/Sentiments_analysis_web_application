import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def train_model():
    """
    Entraîne le modèle sur le dataset Twitter prétraité.
    """
    # Chargement des données
    try:
        df = pd.read_csv('data/twitter_sentiment_processed.csv')
    except FileNotFoundError:
        print("Fichier de données non trouvé. Veuillez d'abord exécuter prepare_twitter_data.py")
        return None
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['target'],
        test_size=0.2,
        random_state=42,
        stratify=df['target']
    )
    
    # Vectorisation
    print("Vectorisation des textes...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)  # Utilisation de bigrammes pour capturer plus de contexte
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Entraînement du modèle
    print("Entraînement du modèle...")
    model = MultinomialNB(alpha=0.1)  # alpha=0.1 pour un peu de lissage
    model.fit(X_train_vec, y_train)
    
    # Évaluation
    print("\nÉvaluation du modèle:")
    y_pred = model.predict(X_test_vec)
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    print(f"\nPrécision globale: {accuracy_score(y_test, y_pred):.3f}")
    
    # Sauvegarde du modèle et du vectoriseur
    print("\nSauvegarde du modèle et du vectoriseur...")
    with open('models/twitter_mnb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/twitter_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Modèle et vectoriseur sauvegardés avec succès!")
    
    return model, vectorizer

if __name__ == "__main__":
    train_model()