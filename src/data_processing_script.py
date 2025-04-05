import pandas as pd

def clean_text(text):
    """Nettoyer un texte : minuscule, enlever les liens et la ponctuation."""
    if pd.isnull(text):
        return ""  # Retourne une chaîne vide si le texte est manquant
    text = text.lower()  # Convertir tout en minuscules
    text = text.replace("http", "")  # Supprimer les liens
    text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Retirer les caractères non alphanumériques
    return text

def preprocess_data(df):
    """Prétraiter les données : nettoyage du texte et ajout d'une colonne 'label'."""
    # Vérifier les premières lignes de la colonne 'content'
    print("Premier aperçu de la colonne 'content' :")
    print(df['content'].head())

    # Convertir la colonne 'score' en numérique, forcer les erreurs à NaN
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Vérifier les premières lignes de la colonne 'score'
    print("Premier aperçu de la colonne 'score' :")
    print(df['score'].head())
    
    # Appliquer la fonction de nettoyage sur chaque texte
    df['text'] = df['content'].apply(clean_text)  # Correction ici: 'review' est remplacé par 'content'

    # S'assurer que le texte est en minuscules
    df['text'] = df['text'].str.lower()

    # Ajouter la colonne 'label' : 1 si score >= 4, sinon 0
    df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else 0)

    # Vérifier les premières lignes du DataFrame traité
    print("Aperçu du DataFrame traité :")
    print(df.head())

    return df[['text', 'label']]

def main():
    column_names = [
        'reviewId', 'userName', 'userImage', 'content', 'score',
        'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent',
        'repliedAt', 'sortOrder', 'appId'
    ]
    
    # Charger le fichier CSV avec détection automatique du séparateur
    df = pd.read_csv('/Users/utilisateur/Projet_Pipeline_Sentiment/data/dataset.csv', header=0, names=column_names, on_bad_lines='skip')
    
    # Vérifiez les colonnes
    print("Colonnes dans le fichier après correction:", df.columns)
    
    # Prétraiter les données
    df_processed = preprocess_data(df)
    print("Data preprocessing completed successfully.")
    print(df_processed.head())  # Affiche les premières lignes du DataFrame traité pour vérification
    
    # Sauvegarder les données traitées dans un fichier CSV
    df_processed.to_csv('data/processed_dataset.csv', index=False)
    print("Données traitées sauvegardées dans 'data/processed_dataset.csv'.")

if __name__ == "__main__":
    main()
