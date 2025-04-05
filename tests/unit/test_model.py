import unittest
from src.model import train_model
from src.data_processing import preprocess_data
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        # Charger un jeu de données d'exemple
        df = pd.read_csv("data/processed_dataset.csv")
        self.df = preprocess_data(df)  # Prétraiter les données
        self.model, self.vectorizer = train_model(self.df)  # Entraîner le modèle

    def test_model_accuracy(self):
        # Vérifier l'exactitude du modèle
        self.assertGreater(self.model.score(self.df['text'], self.df['label']), 0.7)

if __name__ == "__main__":
    unittest.main()
