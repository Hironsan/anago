import os
import time
import unittest

import numpy as np

from anago.utils import load_data_and_labels
from anago.preprocess import StaticPreprocessor, DynamicPreprocessor, UNK, pad_char
from anago.preprocess import StaticPreprocessor


class TestStaticPreprocessor(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_preprocessor(self):
        X, y = load_data_and_labels(self.filename)
        p = StaticPreprocessor()
        p = p.fit(X, y)
        X, y = p.transform(X, y)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(p.inverse_transform(y), list)
        self.assertIsInstance(p.inverse_transform(y)[0], list)
        self.assertIsInstance(p.inverse_transform(y)[0][0], str)


class WordPreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_preprocessor(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = StaticPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(p.inverse_transform(y), list)

    def test_transform_only_words(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = StaticPreprocessor()
        p = preprocessor.fit(X, y)
        X = p.transform(X)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)

    def test_unknown_word(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = StaticPreprocessor()
        p = preprocessor.fit(X, y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = p.transform(X, y)
        print(X)

    def test_vocab_init(self):
        X, y = load_data_and_labels(self.filename)
        unknown_word = 'unknownword'
        X_test, y_test = [[unknown_word]], [['O']]

        preprocessor = StaticPreprocessor()
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0]
        self.assertEqual(words, [p.word_dic[UNK]])

        vocab_init = {unknown_word}
        preprocessor = StaticPreprocessor(vocab_init=vocab_init)
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0]
        self.assertNotEqual(words, [p.word_dic[UNK]])

    def test_save(self):
        preprocessor = StaticPreprocessor()
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        preprocessor.save(filepath)
        self.assertTrue(os.path.exists(filepath))
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_load(self):
        X, y = load_data_and_labels(self.filename)
        p = StaticPreprocessor()
        p.fit(X, y)
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        p.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        loaded_p = StaticPreprocessor.load(filepath)
        x_test1, y_test1 = p.transform(X, y)
        x_test2, y_test2 = loaded_p.transform(X, y)
        np.testing.assert_array_equal(x_test1[0], x_test2[0])  # word
        np.testing.assert_array_equal(x_test1[1], x_test2[1])  # char
        np.testing.assert_array_equal(y_test1, y_test2)
        if os.path.exists(filepath):
            os.remove(filepath)


class PreprocessTest(unittest.TestCase):

    def test_pad_char(self):
        sequences = [[[1, 2, 3, 4], [1, 2], [1], [1, 2, 3]],
                     [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3, 4]]]
        expected_seq = [[[1, 2, 3, 4, 0], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0], [1, 2, 3, 0, 0]],
                        [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 2, 3, 4, 0], [0, 0, 0, 0, 0]]]
        padded_seq = pad_char(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)
