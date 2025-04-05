import unittest
import pandas as pd
from src.data_extraction import load_data, clean_text

class TestDataExtraction(unittest.TestCase):

    def test_load_valid_file(self):
        # Test pour vérifier le chargement d'un fichier valide
        df = load_data("data/dataset.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("text", df.columns)
        self.assertIn("label", df.columns)

    def test_load_invalid_file(self):
        # Test pour vérifier qu'une erreur est levée si le fichier n'existe pas
        with self.assertRaises(FileNotFoundError):
            load_data("data/non_existent_file.csv")

    def test_load_invalid_columns(self):
        # Test pour vérifier qu'une erreur est levée si les colonnes attendues sont manquantes
        invalid_csv = "data/invalid_dataset.csv"
        with self.assertRaises(KeyError):
            load_data(invalid_csv)

    def test_clean_text(self):
        # Test pour vérifier que le texte est bien nettoyé
        dirty = "C'est SUPER! 😊 Visitez https://test.com !!!"
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "cest super visitez")

    def test_preprocess_data(self):
        # Test pour vérifier que les données sont bien traitées
        df = pd.DataFrame({"text": ["Hello WORLD!!", "CLEAN   me!!!", " https://url.com "]})
        processed = preprocess_data(df)
        
        # Vérifier que tous les textes sont en minuscules
        self.assertTrue(all(processed['text'].apply(lambda t: t.islower())), "Tous les textes doivent être en minuscules.")
        
        # Vérifier qu'aucun lien ne reste dans les textes
        self.assertTrue(all("http" not in t for t in processed['text']))

if __name__ == '__main__':
    unittest.main()
