import os
import unittest

import numpy as np

from anago.utils import load_data_and_labels
from anago.preprocess import StaticPreprocessor, DynamicPreprocessor, UNK, pad_nested_sequences


class TestStaticPreprocessor(unittest.TestCase):

    def setUp(self):
        self.p = StaticPreprocessor()

    @classmethod
    def setUpClass(cls):
        filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')
        cls.X, cls.y = load_data_and_labels(filename)

    def test_preprocessor(self):
        X, y = self.p.fit_transform(self.X, self.y)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(self.p.inverse_transform(y), list)
        self.assertIsInstance(self.p.inverse_transform(y)[0], list)
        self.assertIsInstance(self.p.inverse_transform(y)[0][0], str)

    def test_transform_only_words(self):
        self.p.fit(self.X, self.y)
        X = self.p.transform(self.X)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)

    def test_unknown_word(self):
        self.p = StaticPreprocessor()
        self.p.fit(self.X, self.y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = self.p.transform(X, y)
        print(X)

    def test_vocab_init(self):
        unknown_word = 'unknownword'
        X_test, y_test = [[unknown_word]], [['O']]

        self.p.fit(self.X, self.y)
        x_pred = self.p.transform(X_test)
        words = x_pred[0]
        self.assertEqual(words, [self.p._word_vocab[UNK]])

        vocab_init = {unknown_word}
        p = StaticPreprocessor(vocab_init=vocab_init)
        p.fit(self.X, self.y)
        X_pred = p.transform(X_test)
        words = X_pred[0]
        self.assertNotEqual(words, [p._word_vocab[UNK]])

    def test_save(self):
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        self.p.save(filepath)
        self.assertTrue(os.path.exists(filepath))
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_load(self):
        self.p.fit(self.X, self.y)
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        self.p.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        loaded_p = StaticPreprocessor.load(filepath)
        x_test1, y_test1 = self.p.transform(self.X, self.y)
        x_test2, y_test2 = loaded_p.transform(self.X, self.y)
        np.testing.assert_array_equal(x_test1[0], x_test2[0])  # word
        np.testing.assert_array_equal(x_test1[1], x_test2[1])  # char
        np.testing.assert_array_equal(y_test1, y_test2)
        if os.path.exists(filepath):
            os.remove(filepath)


class TestPreprocess(unittest.TestCase):

    def test_pad_nested_sequences(self):
        sequences = [[[1, 2, 3, 4], [1, 2], [1], [1, 2, 3]],
                     [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3, 4]]]
        expected_seq = [[[1, 2, 3, 4, 0], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0], [1, 2, 3, 0, 0]],
                        [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 2, 3, 4, 0], [0, 0, 0, 0, 0]]]
        padded_seq = pad_nested_sequences(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)

        sequences = [[[1, 2], [1]]]
        expected_seq = [[[1, 2], [1, 0]]]
        padded_seq = pad_nested_sequences(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)

        sequences = [[[1], []]]
        expected_seq = [[[1], [0]]]
        padded_seq = pad_nested_sequences(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)

        sequences = [[[1]]]
        expected_seq = [[[1]]]
        padded_seq = pad_nested_sequences(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)

        sequences = [[[]]]
        expected_seq = [[[]]]
        padded_seq = pad_nested_sequences(sequences)
        np.testing.assert_equal(padded_seq, expected_seq)
