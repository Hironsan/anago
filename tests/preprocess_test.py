import os
import time
import unittest

import numpy as np

from anago.data import reader
from anago.data.preprocess import WordPreprocessor, UNK, dense_to_one_hot, pad_sequences, _pad_sequences


class WordPreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/train.txt')

    def test_preprocessor(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        chars, words = X[0]
        char, word = chars[0][0], words[0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(p.inverse_transform(y[0])[0], str)

    def test_unknown_word(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = p.transform(X, y)

    def test_vocab_init(self):
        X, y = reader.load_data_and_labels(self.filename)
        unknown_word = 'unknownword'
        X_test, y_test = [[unknown_word]], [['O']]

        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0][1]
        self.assertEqual(words, [p.vocab_word[UNK]])

        vocab_init = {unknown_word}
        preprocessor = WordPreprocessor(vocab_init=vocab_init)
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0][1]
        self.assertNotEqual(words, [p.vocab_word[UNK]])

    def test_save(self):
        preprocessor = WordPreprocessor()
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        preprocessor.save(filepath)
        self.assertTrue(os.path.exists(filepath))
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_load(self):
        X, y = reader.load_data_and_labels(self.filename)
        p = WordPreprocessor()
        p.fit(X, y)
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        p.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        loaded_p = WordPreprocessor.load(filepath)
        self.assertEqual(loaded_p.transform(X, y), p.transform(X, y))
        if os.path.exists(filepath):
            os.remove(filepath)


class PreprocessTest(unittest.TestCase):

    def test_dense_to_onehot(self):
        # 1d vector
        labels = np.array([1, 2, 3])
        labels_one_hot = dense_to_one_hot(labels, num_classes=9)
        for labels in labels_one_hot:
            self.assertEqual(sum(labels), 1)

        # 2d matrix
        labels = np.array([[1, 2, 3],
                           [4, 5, 6]])
        labels_one_hot = dense_to_one_hot(labels, num_classes=9, nlevels=2)
        for labels in labels_one_hot:
            for l in labels:
                self.assertEqual(sum(l), 1)

        # nlevels test
        with self.assertRaises(ValueError):
            labels_one_hot == dense_to_one_hot(labels, num_classes=9, nlevels=3)

    def test_pad_sequences(self):
        # word level padding
        sequences = [[1, 2],
                     [3, 4, 5, 6]]
        expected_seq = [[1, 2, 0, 0],
                        [3, 4, 5, 6]]
        padded_seq, _ = pad_sequences(sequences, pad_tok=0)
        self.assertEqual(padded_seq, expected_seq)

        # char level padding
        sequences = [[[1, 2, 3, 4], [1, 2], [1], [1, 2, 3]],
                     [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3, 4]]]
        expected_seq = [[[1, 2, 3, 4, 0], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0], [1, 2, 3, 0, 0]],
                        [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 2, 3, 4, 0], [0, 0, 0, 0, 0]]]
        padded_seq, _ = pad_sequences(sequences, pad_tok=0, nlevels=2)
        self.assertEqual(padded_seq, expected_seq)

        # nlevels test
        with self.assertRaises(ValueError):
            pad_sequences(sequences, pad_tok=0, nlevels=3)

    def test__pad_sequences(self):
        sequences = [[1, 2],
                     [3, 4, 5, 6]]
        expected_seq = [[1, 2, 0, 0],
                        [3, 4, 5, 6]]
        max_length = max(len(s) for s in sequences)
        padded_seq, _ = _pad_sequences(sequences, pad_tok=0, max_length=max_length)
        self.assertEqual(padded_seq, expected_seq)


class PerformanceTest(unittest.TestCase):
    """
    Measure execution time
    """
    def setUp(self):
        self.start_time = time.time()
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/train.txt')

    def tearDown(self):
        elapsed = time.time() - self.start_time
        print('{}: {:.3f}'.format(self.id(), elapsed))

    def test_data_loading(self):
        X, y = reader.load_data_and_labels(self.filename)

    def test_fit(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)

    def test_transform(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)

    def test_to_numpy_array(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        y = np.asarray(y)

    def test_pad_sequences(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        y, _ = pad_sequences(y, pad_tok=0)
        char_ids, word_ids = zip(*X)
        word_ids = pad_sequences(word_ids, pad_tok=0)
        char_ids = pad_sequences(char_ids, pad_tok=0, nlevels=2)

    def test_dense_to_onehot(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        _, y = p.transform(X, y)
        y, _ = pad_sequences(y, pad_tok=0)
        y = np.asarray(y)
        labels_one_hot = dense_to_one_hot(y, num_classes=len(p.vocab_tag), nlevels=2)

    def test_calc_sequence_lengths(self):
        X, y = reader.load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        _, y = p.transform(X, y)
        y, _ = pad_sequences(y, pad_tok=0)
        y = np.asarray(y)
        labels_one_hot = dense_to_one_hot(y, num_classes=len(p.vocab_tag), nlevels=2)
        y_t = np.argmax(labels_one_hot, -1)
        y_t = y_t.astype(np.int32)
        sequence_lengths = np.argmin(y_t, -1)
