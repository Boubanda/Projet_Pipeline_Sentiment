import pandas as pd
import joblib

def predict_new_data(input_texts):
    # Charger le modèle et le vectoriseur sauvegardés
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Transformer les textes d'entrée avec le vectoriseur
    X_new = vectorizer.transform(input_texts)

    # Faire des prédictions avec le modèle
    predictions = model.predict(X_new)

    return predictions

def main():
    # Exemple de nouvelles données à prédire
    new_data = [
        "This app is great! I love it.",
        "The app doesn't work as expected, very bad experience.",
        "Amazing experience, highly recommend!"
    ]

    # Prédire sur de nouvelles données
    predictions = predict_new_data(new_data)

    # Afficher les résultats des prédictions
    for text, pred in zip(new_data, predictions):
        print(f"Text: {text}\nPredicted Label: {pred}\n")

if __name__ == "__main__":
    main()
