import os
import unittest

from anago.reader import load_data_and_labels, load_word_embeddings
from anago.models import BiLSTMCRF
from anago.preprocess import WordPreprocessor

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

    def test_train(self):
        data_path = os.path.join(DATA_ROOT, 'train.txt')
        X, y = load_data_and_labels(data_path)

        p = WordPreprocessor()
        p.fit(X, y)

        model = BiLSTMCRF(char_vocab_size=len(p.vocab_char),
                          word_vocab_size=len(p.vocab_word),
                          ntags=len(p.vocab_tag))
        model.preprocessor = p

        model.fit(X, y)

    def test_predict(self):
        pass

    def test_save(self):
        pass

    def test_load(self):
        pass
