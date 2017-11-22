import os
import time
import unittest

import numpy as np

from anago.reader import load_data_and_labels
from anago.preprocess import WordPreprocessor, UNK, dense_to_one_hot, pad_sequences, _pad_sequences


class WordPreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_preprocessor(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(p.inverse_transform(y[0])[0], str)

    def test_transform_only_words(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X = p.transform(X)
        words, chars = X
        char, word = chars[0][0][0], words[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)

    def test_transform_with_padding(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=True)
        p = preprocessor.fit(X, y)
        X = p.transform(X)
        words, chars = X
        word, char = words[0][0], chars[0][0][0]
        self.assertIsInstance(int(word), int)
        self.assertIsInstance(int(char), int)

    def test_unknown_word(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = p.transform(X, y)

    def test_vocab_init(self):
        X, y = load_data_and_labels(self.filename)
        unknown_word = 'unknownword'
        X_test, y_test = [[unknown_word]], [['O']]

        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0][1]
        self.assertEqual(words, [p.vocab_word[UNK]])

        vocab_init = {unknown_word}
        preprocessor = WordPreprocessor(vocab_init=vocab_init, padding=False)
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
        X, y = load_data_and_labels(self.filename)
        p = WordPreprocessor()
        p.fit(X, y)
        filepath = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        p.save(filepath)
        self.assertTrue(os.path.exists(filepath))

        loaded_p = WordPreprocessor.load(filepath)
        x_test1, y_test1 = p.transform(X, y)
        x_test2, y_test2 = loaded_p.transform(X, y)
        np.testing.assert_array_equal(x_test1[0], x_test2[0])  # word
        np.testing.assert_array_equal(x_test1[1], x_test2[1])  # char
        np.testing.assert_array_equal(y_test1, y_test2)
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
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/train.txt')

    def tearDown(self):
        elapsed = time.time() - self.start_time
        print('{}: {:.3f}'.format(self.id(), elapsed))

    def test_data_loading(self):
        X, y = load_data_and_labels(self.filename)

    def test_fit(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)

    def test_transform(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)

    def test_to_numpy_array(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=False)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        y = np.asarray(y)

    def test_pad_sequences(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=True)
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)

    def test_calc_sequence_lengths(self):
        X, y = load_data_and_labels(self.filename)
        preprocessor = WordPreprocessor(padding=True)
        p = preprocessor.fit(X, y)
        _, y = p.transform(X, y)
        y_t = np.argmax(y, -1)
        y_t = y_t.astype(np.int32)
        sequence_lengths = np.argmin(y_t, -1)
