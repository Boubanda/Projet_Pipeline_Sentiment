import pandas as pd
from transformers import AutoTokenizer
import re

def clean_text(text):
    """Nettoyer un texte : minuscule, enlever les liens et la ponctuation."""
    # Vérifier si le texte est bien une chaîne de caractères avant de le traiter
    if isinstance(text, str):
        text = text.lower()  # Convertir tout en minuscules
        text = text.replace("http", "")  # Supprimer les liens
        text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Retirer les caractères non alphanumériques
    return text


def preprocess_data(df):
    """Prétraiter les données : nettoyage du texte et ajout d'une colonne 'label'."""
    
    # Convertir la colonne 'score' en numérique, forcer les erreurs à NaN
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Appliquer la fonction de nettoyage sur chaque texte
    df['text'] = df['content'].apply(clean_text)  # Correction ici: 'review' est remplacé par 'content'

    # S'assurer que le texte est en minuscules
    df['text'] = df['text'].str.lower()

    # Ajouter la colonne 'label' : 1 si score >= 4, sinon 0
    df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else 0)

    return df[['text', 'label']]



