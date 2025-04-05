import pandas as pd
from transformers import AutoTokenizer

def load_data(filepath):
    try:
        # Lire le fichier CSV avec séparation par ';' et une bonne gestion des colonnes
        df = pd.read_csv(filepath, sep=';', header=0, encoding='utf-8')
        
        # Vérifier les colonnes présentes dans le fichier
        print("Colonnes disponibles:", df.columns)
        
        # Si les colonnes 'content' et 'score' sont manquantes, lever une erreur
        if 'content' not in df.columns or 'score' not in df.columns:
            raise KeyError("Les colonnes 'content' ou 'score' sont manquantes dans le fichier.")
        
        # Sélectionner les colonnes pertinentes et supprimer les lignes manquantes
        df = df[['content', 'score']].dropna()

        # Renommer la colonne 'content' en 'text'
        df = df.rename(columns={"content": "text"})

        # Créer la colonne 'label' en fonction de la colonne 'score'
        df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else 0)

        # Retourner le DataFrame avec les colonnes 'text' et 'label'
        return df[['text', 'label']]
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement : {e}")

# Exemple d'appel de fonction
# Si vous souhaitez tester la fonction, décommentez ces lignes et fournissez le chemin correct vers votre fichier.
# df = load_data("path_to_your_file.csv")
# print(df.head())
