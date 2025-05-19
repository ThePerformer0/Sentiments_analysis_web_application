# Modèle d'Analyse de Sentiment Twitter V2

Ce dossier contient les fichiers nécessaires pour l'entraînement et l'utilisation d'un modèle d'analyse de sentiment sur des tweets.

## Structure du Projet

- `prepare_twitter_data.py` : Script de prétraitement des données
- `train_twitter_model.py` : Script d'entraînement du modèle
- `twitter_model.pkl` : Modèle entraîné
- `twitter_vectorizer.pkl` : Vectoriseur TF-IDF

## Source des Données

### Dataset Sentiment140
Le modèle a été entraîné sur le dataset Sentiment140, disponible sur Kaggle :
- **Lien** : [Sentiment140 Dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Taille** : 1.6 million de tweets
- **Format** : CSV
- **Colonnes** :
  - target: le sentiment (0 = négatif, 4 = positif)
  - id: l'identifiant du tweet
  - date: la date du tweet
  - flag: la requête
  - user: l'utilisateur
  - text: le texte du tweet

### Téléchargement et Préparation
Le dataset doit être téléchargé manuellement depuis Kaggle et placé dans le dossier `data/` avant l'exécution des scripts. Le fichier `.gitignore` est configuré pour exclure ces fichiers de données volumineux du contrôle de version.

## Processus d'Entraînement

### 1. Prétraitement des Données (`prepare_twitter_data.py`)

Le script de prétraitement effectue les opérations suivantes :
- Téléchargement du dataset Twitter depuis Kaggle (Sentiment140)
- Nettoyage des tweets :
  - Suppression des URLs
  - Suppression des mentions @ et hashtags #
  - Conversion en minuscules
  - Suppression de la ponctuation et des chiffres
- Prétraitement avancé :
  - Tokenisation
  - Suppression des stopwords
  - Lemmatisation
- Équilibrage des classes (nombre égal de tweets positifs et négatifs)

### 2. Entraînement du Modèle (`train_twitter_model.py`)

Le processus d'entraînement comprend :
- Division des données (80% entraînement, 20% test)
- Vectorisation TF-IDF avec les paramètres :
  - `max_features=10000`
  - `min_df=5`
  - `max_df=0.8`
  - `ngram_range=(1, 2)` pour capturer le contexte
- Utilisation d'un classificateur Naive Bayes Multinomial avec `alpha=0.1`

## Limitations du Modèle

### Classification Binaire
Le modèle actuel présente une limitation importante : il ne peut classifier que deux types de sentiments (positif et négatif). Cette limitation est due à :

1. **Structure du Dataset** :
   - Le dataset Sentiment140 utilisé pour l'entraînement est binaire par nature
   - Les tweets sont annotés uniquement comme positifs (4) ou négatifs (0)
   - Aucune classe "neutre" n'est présente dans les données d'entraînement

2. **Architecture du Modèle** :
   - Le classificateur Naive Bayes Multinomial est configuré pour une classification binaire
   - Les paramètres de vectorisation TF-IDF sont optimisés pour cette tâche binaire

### Impact sur l'Utilisation
Cette limitation signifie que :
- Les textes neutres seront forcément classés comme positifs ou négatifs
- Le modèle peut ne pas être optimal pour des textes qui ne contiennent pas de sentiment marqué
- Les résultats doivent être interprétés en tenant compte de cette contrainte binaire

## Évaluation des Performances

### Métriques d'Évaluation

- **Précision globale (Accuracy)**: 0.768 (76.8%)

### Rapport de Classification Détaillé

| Classe    | Precision | Recall | F1-Score | Support |
|-----------|-----------|---------|-----------|----------|
| Negative  | 0.77      | 0.76    | 0.77      | 159,204  |
| Positive  | 0.76      | 0.77    | 0.77      | 159,203  |

### Analyse des Résultats

1. **Équilibre des Classes** :
   - Distribution équilibrée entre les classes positive et négative
   - Support similaire pour les deux classes (159,203 vs 159,204)

2. **Stabilité des Prédictions** :
   - Scores F1 identiques (0.77) pour les deux classes
   - Précision et rappel très proches pour chaque classe

3. **Performance Globale** :
   - Précision globale de 76.8%
   - Moyennes macro et weighted identiques (0.77)

## Utilisation du Modèle

Pour utiliser le modèle :
1. Charger le modèle et le vectoriseur depuis les fichiers .pkl
2. Prétraiter le nouveau texte de la même manière que les données d'entraînement
3. Vectoriser le texte prétraité
4. Utiliser le modèle pour prédire le sentiment

## Améliorations Possibles

1. **Support des Sentiments Neutres** :
   - Intégration d'un nouveau dataset incluant des sentiments neutres
   - Modification de l'architecture pour supporter la classification multi-classes
   - Ajustement des hyperparamètres pour gérer trois classes

2. **Augmentation des Données** :
   - Utilisation de techniques d'augmentation de données
   - Intégration de données supplémentaires

3. **Optimisation du Modèle** :
   - Test d'autres algorithmes (SVM, BERT, etc.)
   - Optimisation des hyperparamètres

4. **Prétraitement** :
   - Amélioration de la gestion des emojis
   - Meilleure gestion du langage informel 