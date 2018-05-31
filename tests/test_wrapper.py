import os
import unittest
from pprint import pprint

import numpy as np

import anago
from anago.utils import load_data_and_labels


def load_glove(file):
    """Loads GloVe vectors in numpy array.

    Args:
        file (str): a path to a glove file.

    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file) as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model

get_path = lambda path: os.path.join(os.path.dirname(__file__), path)
DATA_ROOT = get_path('../data/conll2003/en/ner')
SAVE_ROOT = get_path('models')  # trained model
LOG_ROOT = get_path('logs')     # checkpoint, tensorboard
EMBEDDING_PATH = get_path('../data/glove.6B/glove.6B.100d.txt')


class TestWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        test_path = os.path.join(DATA_ROOT, 'test.txt')

        cls.x_train, cls.y_train = load_data_and_labels(train_path)
        cls.x_valid, cls.y_valid = load_data_and_labels(valid_path)
        cls.x_test, cls.y_test = load_data_and_labels(test_path)

        cls.embeddings = load_glove(EMBEDDING_PATH)
        cls.words = 'President Obama is speaking at the White House.'.split()
        cls.dir_path = 'models'

    def test_train_without_pretrained_embedding(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def test_train_with_pretrained_embedding(self):
        model = anago.Sequence(embeddings=self.embeddings)
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def test_eval(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)
        model.score(self.x_test, self.y_test)

    def test_analyze(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)
        res = model.analyze(self.words)
        pprint(res)

        self.assertIn('words', res)
        self.assertIn('entities', res)

    def test_save_and_load(self):
        weights_file = ''
        params_file = ''
        preprocessor_file = ''

        model = anago.Sequence()
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid)
        model.save(weights_file, params_file, preprocessor_file)
        score1 = model.score(self.x_test, self.y_test)

        self.assertTrue(weights_file)
        self.assertTrue(params_file)
        self.assertTrue(preprocessor_file)

        model = anago.Sequence.load(weights_file, params_file, preprocessor_file)
        score2 = model.score(self.x_test, self.y_test)

        self.assertEqual(score1, score2)

    def test_train_vocab_init(self):
        vocab = set()
        for words in np.r_[self.x_train, self.x_valid, self.x_test]:
            for word in words:
                vocab.add(word)
        model = anago.Sequence(initial_vocab=vocab)
        model.fit(self.x_train, self.y_train, self.x_test, self.y_test)
