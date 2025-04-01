import unittest
import pandas as pd
from src.data_extraction import load_data

class TestDataExtraction(unittest.TestCase):
    def test_load_valid_file(self):
        # Chargement du fichier de test
        df = load_data("data/dataset.csv")

        # Vérifie que c'est bien un DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Vérifie la présence des colonnes attendues
        self.assertIn("text", df.columns)
        self.assertIn("label", df.columns)

        # Test bonus : s'assurer que le dataframe n'est pas vide
        self.assertGreater(len(df), 0, "Le DataFrame est vide")

if __name__ == "__main__":
    unittest.main()
