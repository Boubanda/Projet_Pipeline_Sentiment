import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')

        df = df[['content', 'score']].dropna()
        df = df.rename(columns={"content": "text"})

        # Conversion explicite des scores en float
        df['score'] = pd.to_numeric(df['score'], errors='coerce')  # transforme les valeurs non valides en NaN
        df = df.dropna(subset=['score'])  # on supprime les lignes avec des scores non convertibles

        df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else 0)
        return df[['text', 'label']]

    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement : {e}")
