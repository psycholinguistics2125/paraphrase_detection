import unittest
from src.paraphrase_generation import generate_paraphrase_from_model_name  # Import your code
import yaml

class TestParaphraseGenerator(unittest.TestCase):
    def setUp(self):
        with open("config.yaml", "r") as config_file:
            self.config = yaml.safe_load(config_file)['paraphrase_dataset']

    def test_generate_paraphrase(self):
        model_name = self.config["paraphrase_generator"]["default_model"]
        sentence = "Input sentence to paraphrase."
        paraphrase = generate_paraphrase_from_model_name(sentence, model_name)
        print(paraphrase)
        self.assertIsInstance(paraphrase, str)
        self.assertTrue(paraphrase)  # Ensure the paraphrase is not an empty string

    def test_generate_paraphrase_with_custom_model(self):
        model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        sentence = "Input sentence to paraphrase."
        paraphrase = generate_paraphrase_from_model_name(sentence, model_name)
        self.assertIsInstance(paraphrase, str)
        self.assertTrue(paraphrase)  # Ensure the paraphrase is not an empty string

if __name__ == "__main__":
    unittest.main()