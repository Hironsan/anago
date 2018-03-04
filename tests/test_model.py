import os
import unittest

from anago.utils import load_data_and_labels
from anago.models import BiLSTMCRF

get_path = lambda path: os.path.join(os.path.dirname(__file__), path)
DATA_ROOT = get_path('../data/conll2003/en/ner')
SAVE_ROOT = get_path('models')  # trained model


class TrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

    def setUp(self):
        data_path = os.path.join(DATA_ROOT, 'train.txt')
        self.X, self.y = load_data_and_labels(data_path)

    def test_build_model(self):
        model = BiLSTMCRF(char_vocab_size=100,
                          word_vocab_size=10000,
                          ntags=10)
        model.build_model()

    def test_predict(self):
        pass

    def test_save(self):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        model = BiLSTMCRF(char_vocab_size=100,
                          word_vocab_size=10000,
                          ntags=10)
        model.build_model()

        self.assertFalse(os.path.exists(weights_file))
        self.assertFalse(os.path.exists(params_file))

        model.save(weights_file, params_file)

        self.assertTrue(os.path.exists(weights_file))
        self.assertTrue(os.path.exists(params_file))

        os.remove(weights_file)
        os.remove(params_file)

    def test_load(self):
        pass
