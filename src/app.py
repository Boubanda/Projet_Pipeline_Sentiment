from flask import Flask, request, jsonify
import joblib

# Charger le modèle et le vectoriseur
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Récupérer les données envoyées en JSON
    text = data.get('text', '')  # Extraire le texte du JSON
    
    # Transformer le texte en vecteurs
    text_vect = vectorizer.transform([text])
    
    # Prédiction du modèle
    prediction = model.predict(text_vect)
    
    # Convertir la prédiction en un type JSON sérialisable (par exemple, int)
    prediction = int(prediction[0])  # Convertir en int (valeur Python native)

    return jsonify({'prediction': prediction})  # Retourner la prédiction en JSON

# Lancer l'application
if __name__ == '__main__':
    # Exécuter l'application Flask en écoutant sur 0.0.0.0 pour qu'elle soit accessible depuis l'extérieur du conteneur Docker
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # logique de prédiction ici
    return {"prediction": 1}

if __name__ == "__main__":
    app.run()
