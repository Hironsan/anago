import os
import unittest

from anago.reader import load_data_and_labels
from anago.models import BiLSTMCRF
from anago.preprocess import StaticPreprocessor, DynamicPreprocessor

get_path = lambda path: os.path.join(os.path.dirname(__file__), path)
DATA_ROOT = get_path('../data/conll2003/en/ner')
SAVE_ROOT = get_path('models')  # trained model
LOG_ROOT = get_path('logs')     # checkpoint, tensorboard
EMBEDDING_PATH = get_path('../data/glove.6B/glove.6B.100d.txt')


class TrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

    def setUp(self):
        data_path = os.path.join(DATA_ROOT, 'train.txt')
        self.X, self.y = load_data_and_labels(data_path)

    def test_train(self):
        p = StaticPreprocessor()
        p.fit(self.X, self.y)
        X, y = p.transform(self.X, self.y)

        model = BiLSTMCRF(char_vocab_size=len(p.char_dic),
                          word_vocab_size=len(p.word_dic),
                          ntags=len(p.label_dic))
        dp = DynamicPreprocessor(n_labels=len(p.label_dic))
        model.preprocessor = dp
        model.fit(X, y)

    def test_predict(self):
        pass

    def test_save(self):
        pass

    def test_load(self):
        pass
