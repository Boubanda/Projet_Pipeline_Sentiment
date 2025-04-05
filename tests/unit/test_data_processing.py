import unittest
import pandas as pd
from src.data_processing import clean_text, preprocess_data

class TestDataProcessing(unittest.TestCase):

    def test_clean_text(self):
        dirty = "C'est SUPER! 😊 Visitez https://test.com !!!"
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "cest super visitez")

    def test_preprocess_data(self):
        df = pd.DataFrame({"text": ["Hello WORLD!!", "CLEAN   me!!!", " https://url.com "]})
        processed = preprocess_data(df)
        
        # Vérifier que tous les textes sont en minuscules
        for text in processed['text']:
            self.assertTrue(text.islower(), f"Text: {text} n'est pas en minuscules.")
        
        # Vérifier que toutes les URLs ont été supprimées
        self.assertTrue(all("http" not in t for t in processed['text']))

    def test_clean_text_empty_input(self):
        dirty = ""
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "")

if __name__ == '__main__':
    unittest.main()
