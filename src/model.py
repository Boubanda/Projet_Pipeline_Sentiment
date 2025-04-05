# model.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df):
    """
    Entraîne un modèle de régression logistique pour la classification de texte.
    
    Args:
    - df (DataFrame): Le DataFrame contenant les données traitées, avec les colonnes 'text' et 'label'.
    
    Returns:
    - model: Le modèle entraîné (LogisticRegression).
    - vectorizer: Le CountVectorizer utilisé pour transformer le texte.
    """
    
    # Diviser les données en variables indépendantes (X) et cibles (y)
    X = df['text']  # Le texte
    y = df['label']  # Les labels (0 ou 1)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser le CountVectorizer pour transformer le texte en vecteurs de caractéristiques
    vectorizer = CountVectorizer(stop_words='english')

    # Transformer les données textuelles en matrices de caractéristiques
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Initialiser le modèle de régression logistique
    model = LogisticRegression()

    # Entraîner le modèle
    model.fit(X_train_vect, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test_vect)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return model, vectorizer

def save_model(model, filename="model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    
    Args:
    - model: Le modèle à sauvegarder.
    - filename: Le nom du fichier pour sauvegarder le modèle.
    """
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename="model.pkl"):
    """
    Charge un modèle depuis un fichier.
    
    Args:
    - filename: Le nom du fichier contenant le modèle sauvegardé.
    
    Returns:
    - model: Le modèle chargé.
    """
    import pickle
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
