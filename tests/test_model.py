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
        weight_file = os.path.join(SAVE_ROOT, 'weights.h5')
        param_file = os.path.join(SAVE_ROOT, 'hyperparameters.h5')
        model = BiLSTMCRF(char_vocab_size=100,
                          word_vocab_size=10000,
                          ntags=10)
        model.build_model()

        self.assertFalse(os.path.exists(weight_file))
        self.assertFalse(os.path.exists(param_file))

        model.save_weights(weight_file)
        model.save_params(param_file)

        self.assertTrue(os.path.exists(weight_file))
        self.assertTrue(os.path.exists(param_file))

        os.remove(weight_file)
        os.remove(param_file)

    def test_load(self):
        pass
