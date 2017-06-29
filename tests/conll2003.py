import os
import unittest
from itertools import chain


from anago.data.conll2003 import load_file, load_data


class TestConll2003(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/test.txt')

    def test_load_file(self):
        words, pos_tags, chunk_tags, ne_tags = load_file(self.filename)
        self.assertEqual(len(words), len(ne_tags))
        self.assertNotEqual(words, [])

    def test_load_data(self):
        X_words_train, y_train, X_words_test, y_test, index2word, index2chunk = load_data()
        self.assertEqual(len(X_words_train), len(y_train))
        self.assertEqual(len(X_words_test), len(y_test))

        words = {word for word in chain(*X_words_train)}
        self.assertNotEqual(len(words), len(index2word))
        self.assertEqual(len(words)+2, len(index2word))  # considering <UNK>, <PAD>

        with open(self.filename) as f:
            lines = [line.strip() for line in f if not line.startswith('-DOCSTART-')]
            lines = [line for line in lines if line != '']
        words = [word for word in chain(*X_words_test)]
        self.assertEqual(len(lines), len(words))

        tags = [tag for tag in chain(*y_train)]
        self.assertTrue(all(isinstance(word, int) for word in words))
        self.assertTrue(all(isinstance(tag, int) for tag in tags))
