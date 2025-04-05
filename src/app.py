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
    
    # Transformer le texte
    text_vect = vectorizer.transform([text])
    
    # Prédiction
    prediction = model.predict(text_vect)
    
    # Convertir la prédiction en un type JSON sérialisable (par exemple, int)
    prediction = int(prediction[0])  # Convertir en int (valeur Python native)

    return jsonify({'prediction': prediction})  # Retourner la prédiction en JSON

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)
