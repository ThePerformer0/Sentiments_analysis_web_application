import streamlit as st
import pickle
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Téléchargement silencieux des ressources NLTK nécessaires
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment Twitter",
    page_icon="😊",
    layout="centered"
)

# Titre et description
st.title("📊 Analyse de Sentiment Twitter")
st.markdown("""
Cette application utilise un modèle de Machine Learning (Naive Bayes) entraîné sur plus d'un million de tweets
pour analyser le sentiment d'un texte. Le modèle peut prédire si un texte est :
- 😊 Positif
- 😞 Négatif
""")

def clean_text(text):
    """Nettoie le texte en entrée"""
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

def preprocess_text(text):
    """Prétraite le texte en entrée"""
    if not isinstance(text, str):
        return ""
    
    # Vérifie et télécharge 'punkt' si besoin
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
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

# Chargement du modèle et du vectoriseur
@st.cache_resource
def load_model():
    with open('models/twitter_mnb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/twitter_tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Interface utilisateur
text_input = st.text_area("Entrez votre texte ici :", height=150)

if st.button("Analyser le sentiment"):
    if text_input:
        # Prétraitement du texte
        cleaned_text = clean_text(text_input)
        processed_text = preprocess_text(cleaned_text)
        
        # Chargement du modèle
        model, vectorizer = load_model()
        
        # Prédiction
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        
        # Affichage du résultat
        st.markdown("---")
        
        # Définition des émojis et couleurs selon le sentiment
        sentiment_emojis = {
            'positive': '😊',
            'negative': '😞'
        }
        
        sentiment_colors = {
            'positive': 'green',
            'negative': 'red'
        }
        
        # Affichage du résultat principal
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {sentiment_colors[prediction]}20;'>
            <h2 style='margin: 0;'>{sentiment_emojis[prediction]} Sentiment : {prediction.upper()} {sentiment_emojis[prediction]}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Section explicative
with st.expander("ℹ️ Comment fonctionne cette application ?"):
    st.markdown("""
    ### Le modèle derrière l'application
    
    Cette application utilise un modèle Naive Bayes Multinomial entraîné sur plus d'un million de tweets. Voici comment il fonctionne :
    
    1. **Prétraitement du texte** :
       - Le texte est nettoyé (suppression des URLs, mentions, hashtags, etc.)
       - Les mots sont lemmatisés (réduits à leur racine)
       - Les stopwords sont supprimés
    
    2. **Vectorisation** :
       - Le texte est transformé en vecteurs numériques via TF-IDF
       - Utilisation de bigrammes pour capturer plus de contexte
       - Vocabulaire de 10,000 termes les plus pertinents
    
    3. **Prédiction** :
       - Le modèle Naive Bayes utilise ces vecteurs pour prédire le sentiment
       - Il peut classer le texte en deux catégories : positif ou négatif
    
    ### Performance du modèle
    
    Le modèle atteint une précision d'environ 80% sur l'ensemble de test, ce qui est un excellent résultat pour une tâche de classification de sentiment.
    """) 