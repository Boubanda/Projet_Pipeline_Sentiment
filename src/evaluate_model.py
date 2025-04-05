import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Charger le modèle et le vectoriseur
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Charger les données de test
df_test = pd.read_csv('data/test_dataset.csv')  # Remplacez ce fichier par votre jeu de test

# Nettoyer les données de test (remplacer NaN par une chaîne vide)
df_test['text'] = df_test['text'].fillna('')

# Transformer les textes en vecteurs
X_test_vect = vectorizer.transform(df_test['text'])

# Obtenir les vraies étiquettes
y_test = df_test['label']

# Prédictions du modèle
y_pred = model.predict(X_test_vect)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur le jeu de test : {accuracy:.4f}")


import joblib

# Charger le modèle et le vectoriseur
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Exemple de nouveau texte pour prédiction
new_text = ["I am happy with this product!"]  # Remplacez cette phrase par un texte quelconque

# Transformer le texte en vecteurs
new_text_vect = vectorizer.transform(new_text)

# Prédire la classe du texte
predicted_label = model.predict(new_text_vect)

print(f"Prédiction pour le texte '{new_text[0]}': {predicted_label[0]}")
