# 🩺 Diabetes Prediction Using Logistic Regression with PySpark

## 1. Présentation du projet

Ce projet consiste à développer un modèle de Machine Learning basé sur la **régression logistique (Logistic Regression)** afin de prédire la présence ou l'absence du diabète chez des individus à partir de données médicales.

L'objectif est d'utiliser les techniques de traitement et d'analyse des données avec **Apache Spark (PySpark)** pour préparer les données, sélectionner les variables pertinentes, entraîner un modèle de classification et évaluer ses performances.

La régression logistique est un algorithme d'apprentissage supervisé particulièrement adapté aux problèmes de classification binaire.

## 2. Objectifs du projet

Les principaux objectifs sont :

- Explorer et comprendre le jeu de données relatif au diabète.
- Identifier et traiter les valeurs manquantes.
- Réaliser une analyse statistique descriptive des variables.
- Sélectionner les caractéristiques pertinentes (Feature Selection).
- Normaliser les variables numériques.
- Construire un modèle de régression logistique avec PySpark.
- Évaluer les performances du modèle de classification.
- Analyser la capacité du modèle à prédire le diabète.

## 3. Technologies et bibliothèques utilisées

Le projet utilise les technologies et bibliothèques Python suivantes :

| Bibliothèque | Utilisation |
|---|---|
| PySpark | Traitement et manipulation des données distribuées |
| Spark SQL | Manipulation des DataFrames |
| NumPy | Calculs numériques |
| Pandas | Manipulation et analyse des données |
| Matplotlib | Visualisation des données |
| Seaborn | Visualisation statistique |
| PySpark MLlib | Analyse statistique |
| PySpark ML | Préparation des caractéristiques et Machine Learning |

### Importation des bibliothèques

```python
from pyspark.sql import SparkSession

# PySpark Functions
from pyspark.sql.functions import (
    when, col, sum, count, isnan, udf, mean
)
import pyspark.sql.functions as F

# Statistical Analysis
from pyspark.mllib.stat import Statistics

# Feature Engineering
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# Model Evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Logistic Regression
from pyspark.ml.classification import LogisticRegression

# Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
```

## 4. Description du jeu de données

Le projet utilise un jeu de données médicales sur le diabète.

Les variables explicatives peuvent notamment inclure :

- **Pregnancies** : nombre de grossesses.
- **Glucose** : concentration de glucose dans le sang.
- **BloodPressure** : pression artérielle.
- **SkinThickness** : épaisseur du pli cutané.
- **Insulin** : taux d'insuline.
- **BMI** : indice de masse corporelle.
- **DiabetesPedigreeFunction** : indicateur lié aux antécédents familiaux.
- **Age** : âge de l'individu.
- **Outcome** : variable cible indiquant la présence ou l'absence du diabète.

La variable cible `Outcome` représente :

- `0` : absence de diabète.
- `1` : présence de diabète.

## 5. Méthodologie et étapes du projet

Le projet est organisé autour des étapes suivantes :

### Étape 1 : Initialisation de Spark

Création d'une session Spark pour permettre le traitement des données avec PySpark.

### Étape 2 : Chargement des données

Importation du dataset et création d'un DataFrame Spark.

### Étape 3 : Exploration des données (EDA)

Analyse exploratoire permettant de comprendre la structure et les caractéristiques du dataset.

Cette étape comprend :
- L'affichage des premières observations.
- L'analyse du schéma des données.
- Le calcul des statistiques descriptives.
- L'analyse de la distribution des variables.
- L'identification des valeurs manquantes.

### Étape 4 : Nettoyage des données

Préparation des données avant la modélisation :

- Détection des valeurs nulles.
- Identification des valeurs aberrantes.
- Traitement des valeurs manquantes.
- Vérification des types de données.

### Étape 5 : Feature Engineering

Transformation des variables explicatives en vecteur de caractéristiques à l'aide de `VectorAssembler`.

Cette étape permet de regrouper les variables indépendantes dans une seule colonne appelée `features`.

### Étape 6 : Normalisation des données

Utilisation de `MinMaxScaler` pour ramener les variables numériques dans un intervalle compris entre 0 et 1.

Cette transformation permet d'harmoniser les échelles des variables avant l'entraînement du modèle.

### Étape 7 : Séparation des données

Division du dataset en deux ensembles :

- **Training Set** : données utilisées pour entraîner le modèle.
- **Testing Set** : données utilisées pour évaluer les performances du modèle.

### Étape 8 : Entraînement du modèle

Construction et entraînement d'un modèle de régression logistique avec PySpark ML.

L'algorithme apprend les relations entre les variables médicales et la variable cible `Outcome`.

### Étape 9 : Prédiction

Utilisation du modèle entraîné pour prédire la classe des nouvelles observations.

Les prédictions correspondent à deux catégories :

- 0 : Non diabétique.
- 1 : Diabétique.

### Étape 10 : Évaluation du modèle

Évaluation des performances à l'aide de plusieurs métriques de classification :

- Accuracy (Exactitude).
- Precision (Précision).
- Recall (Rappel).
- F1-Score.
- Confusion Matrix (Matrice de confusion).

L'objectif est de mesurer la capacité du modèle à identifier correctement les individus diabétiques et non diabétiques.

## 6. Algorithme utilisé : Logistic Regression

La régression logistique est un algorithme de classification supervisée permettant d'estimer la probabilité qu'une observation appartienne à une classe donnée.

Dans ce projet, elle est utilisée pour résoudre un problème de classification binaire.

Le modèle estime la probabilité de présence du diabète à partir des caractéristiques médicales des individus.

La classe prédite dépend ensuite d'un seuil de décision, généralement fixé à 0.5.

## 7. Résultats attendus

À la fin du projet, le modèle doit permettre :

- De prédire la présence du diabète à partir des données médicales.
- D'identifier les variables ayant une influence sur les prédictions.
- D'évaluer les performances de la régression logistique.
- De mesurer les erreurs de classification.
- De démontrer l'utilisation de PySpark pour un projet de Machine Learning.

**Remarque :** les performances réelles du modèle seront renseignées après son entraînement et son évaluation.

## 8. Structure du projet

```text
Diabetes-Prediction-Logistic-Regression/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── diabetes_logistic_regression.ipynb
│
├── README.md
│
└── requirements.txt
```

## 9. Installation et exécution

### Prérequis

- Python 3.x
- Apache Spark
- Java (version compatible avec Spark)
- PySpark
- Jupyter Notebook

### Installation des dépendances

```bash
pip install pyspark numpy pandas matplotlib seaborn jupyter
```

### Lancement du projet

Cloner le dépôt GitHub :

```bash
git clone https://github.com/username/Diabetes-Prediction-Logistic-Regression.git
```

Accéder au répertoire :

```bash
cd Diabetes-Prediction-Logistic-Regression
```

Lancer Jupyter Notebook :

```bash
jupyter notebook
```

Ouvrir ensuite le notebook :

`notebooks/diabetes_logistic_regression.ipynb`

## 10. Conclusion

Ce projet permet de mettre en pratique les principales étapes d'un processus de Machine Learning, depuis l'exploration et le nettoyage des données jusqu'à l'entraînement et l'évaluation d'un modèle de classification.

L'utilisation de PySpark permet également de découvrir les outils de traitement distribué des données et les fonctionnalités de Machine Learning proposées par Spark ML.

Ce travail constitue une application pratique de la régression logistique dans le domaine de la santé et de l'analyse prédictive.

**Avertissement :** ce modèle est réalisé dans un cadre pédagogique. Il ne constitue pas un outil de diagnostic médical et ne doit pas être utilisé pour prendre des décisions cliniques.
