# Analyse de Sentiment sur des Commentaires Textuels

Projet réalisé dans le cadre du cours d’informatique décisionnel de l’organisation **INFO-DECISIO**.  
L’objectif : construire un pipeline robuste pour prédire le **sentiment** (positif / neutre / négatif) de commentaires issus des réseaux sociaux, en s’appuyant sur les techniques modernes d’analyse de données et de machine learning.

---

## 📋 Contexte & Objectifs

L’analyse automatique des opinions (sentiment analysis) est cruciale pour de nombreux métiers : veille, réputation, marketing, etc.  
Ce projet vise à :

- Déployer un pipeline complet : exploration, nettoyage, vectorisation, modélisation, évaluation et sauvegarde.
- Rendre le processus reproductible et facilement intégrable (préparation à une application web).
- Sélectionner le modèle le plus performant et justifier ce choix selon les résultats obtenus.

---

## 📦 Données

- **Source** : [Kaggle – Sentiment Analysis EDA and Prediction (input)](https://www.kaggle.com/code/alokkumar2507/sentiment-analysis-eda-and-prediction/input)
- **Fichier** : `data/sentiment_analysis.csv`
  > **NB :** Télécharger manuellement le jeu de données sur Kaggle et le placer dans le dossier `data/` à la racine du projet.

---

## 🗂️ Structure du projet

- `notebooks/1_data_exploration.ipynb` : Analyse exploratoire (EDA)
- `notebooks/2_data_preprocessing.ipynb` : Nettoyage et vectorisation des textes
- `notebooks/3_model_training_evaluation.ipynb` : Modélisation et évaluation
- `data/` : Contient le dataset
- `models/` : Contient les modèles et vectoriseurs sauvegardés
- `README.md` : Ce document

---

## 🚦 Pipeline du projet

### 1. Analyse exploratoire des données (EDA)

**Objectifs** :
- Comprendre la structure et la qualité du dataset
- Analyser la distribution des sentiments
- Identifier les valeurs manquantes
- Explorer les caractéristiques textuelles (longueur, mots fréquents)
- Visualiser via graphiques et nuages de mots

---

### 2. Prétraitement des données textuelles

**Étapes clés** :
- **Nettoyage** : conversion en minuscules, suppression des URLs, mentions, hashtags, chiffres et ponctuations.
- **Tokenisation & Lemmatisation** (NLTK) : segmentation en mots, suppression des stopwords anglais, réduction à la racine.
- **Vectorisation** : transformation en vecteurs numériques via **TF-IDF** (maximum 5000 termes, min_df=5, max_df=0.8).
- **Split train/test** : division 80/20, stratification sur la cible pour respecter la répartition des classes.

**Exemple de distribution après split** :
- **Train** : neutral ~40%, positive ~33%, negative ~27%
- **Test** : proportions similaires

---

### 3. Modélisation et évaluation

**Modèles testés** :
- **Naive Bayes Multinomial** : simple et efficace pour du texte
- **Régression Logistique** : baseline linéaire interprétable
- **SVM linéaire (LinearSVC)** : performant en haute dimension (texte vectorisé)

**Métriques utilisées** :
- Accuracy
- Precision, Recall, F1-score (moyenne pondérée & par classe)
- Matrice de confusion détaillée
- Analyse du déséquilibre entre classes

**Extrait de résultats (test set)** :

| Modèle                  | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------------------------|----------|----------------------|-------------------|---------------|
| Naive Bayes Multinomial | 0.6200   | 0.6359               | 0.6200            | 0.6124        |
| Régression Logistique   | 0.6400   | 0.6601               | 0.6400            | 0.6309        |
| Linear SVM              | **0.6500**   | 0.6493               | **0.6500**            | **0.6430**        |

> - **Linear SVM** > meilleure accuracy & F1 global.
> - Tous les modèles ont du mal sur la classe "negative", mais Linear SVM équilibre mieux les prédictions et diminue le biais sur la classe "neutral".
> - La Régression Logistique est très précise sur "positive" mais plus biaisée vers "neutral".

---

### 4. Sélection et justification du modèle final

Après analyse :

- **Linear SVM** est retenu pour :
  - Sa meilleure performance globale (accuracy & F1 pondéré)
  - Son rappel "negative" légèrement meilleur
  - Moins de biais vers la classe majoritaire
  - Rapidité et compacité pour une intégration web future

---

### 5. Sauvegarde et perspectives d’intégration

- **Le modèle SVM entraîné** et le **vectoriseur TF-IDF** sont sauvegardés dans `models/` (`linear_svc_model.pkl`, `tfidf_vectorizer.pkl`) via `pickle` pour un futur déploiement (API ou app web).
- **Prochaines pistes** :
  - Tunings d’hyperparamètres
  - Tests avec d’autres méthodes de vectorisation (embeddings)
  - Expérimentations avec des architectures avancées (deep learning)

---

## ⚙️ Prérequis & installation

- Python ≥ 3.7
- Jupyter Notebook
- Bibliothèques principales :  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `wordcloud`

**Installation rapide** :
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

**Pour NLTK, télécharger les ressources nécessaires** :
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 Reproduire l’expérience

1. **Télécharge le dataset** sur Kaggle et place-le dans `data/sentiment_analysis.csv`
2. **Ouvre les notebooks dans l’ordre** :
   - `notebooks/1_data_exploration.ipynb`
   - `notebooks/2_data_preprocessing.ipynb`
   - `notebooks/3_model_training_evaluation.ipynb`
3. **Exécute chaque cellule** pour suivre le pipeline complet
4. **Les modèles et vectoriseurs** sont sauvegardés dans `models/` à la fin du notebook 3

---

## 📊 Résultats attendus

- Visualisations détaillées sur la structure et la répartition des données
- Texte nettoyé, vectorisé et prêt pour la modélisation
- Comparatif objectif des modèles classiques
- Sélection et justification argumentée du meilleur modèle
- Prêt pour une intégration (API, app web, etc.)

---

## ✍️ Auteur

- [ThePerformer0](https://github.com/ThePerformer0)  
Projet développé dans l’organisation **INFO-DECISIO** pour le cours d’informatique décisionnel.

---

## 🙏 Remarques & extensions possibles

- Ce projet constitue une base solide pour expérimenter avec des modèles avancés (Word Embeddings, deep learning, transformers, gestion du déséquilibre des classes…)
- N’hésite pas à cloner le repo, tester sur d’autres jeux de données ou proposer des améliorations/pull requests !

---

**Bon apprentissage et bonne exploration du sentiment !**
