import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    # Charger le fichier CSV avec les données traitées
    df = pd.read_csv('data/processed_dataset.csv')

    # Remplacer les NaN par une chaîne vide ou les supprimer
    df['text'] = df['text'].fillna('')  # Solution 1 : Remplacer NaN par une chaîne vide
    # df = df.dropna(subset=['text'])  # Solution 2 : Supprimer les lignes avec NaN

    # Vérification des données
    print("Aperçu des données :")
    print(df.head())

    # Séparer les features (X) et la cible (y)
    X = df['text']
    y = df['label']

    # Initialisation du vectoriseur
    vectorizer = TfidfVectorizer()

    # Transformation des textes
    X_vect = vectorizer.fit_transform(X)

    # Entraînement du modèle
    print("Entraînement du modèle...")
    model = LogisticRegression()
    model.fit(X_vect, y)

    # Évaluation du modèle
    accuracy = model.score(X_vect, y)
    print(f"Accuracy: {accuracy:.4f}")

    # Sauvegarde du modèle et du vectoriseur
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Modèle entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    main()
