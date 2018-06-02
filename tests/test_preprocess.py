import os
import shutil
import unittest

import numpy as np

from anago.preprocessing import IndexTransformer, pad_nested_sequences


class TestIndexTransformer(unittest.TestCase):

    def setUp(self):
        self.x = [['a'], ['aa', 'ab'], ['AA', 'ab', 'ac']]
        self.y = [['O'], ['B-A', 'I-A'], ['O', 'O', 'B-A']]

    @classmethod
    def setUpClass(cls):
        cls.save_root = os.path.join(os.path.dirname(__file__), 'data')
        cls.preprocessor_file = os.path.join(cls.save_root, 'preprocessor.pkl')
        if not os.path.exists(cls.save_root):
            os.mkdir(cls.save_root)
        if os.path.exists(cls.preprocessor_file):
            os.remove(cls.preprocessor_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_root)

    def test_vocab_size_lower_on(self):
        word_vocab_size = 4
        char_vocab_size = 4
        label_size = 3

        # lower is effective.
        it = IndexTransformer(lower=True)
        it.fit(self.x, self.y)
        self.assertEqual(it.word_vocab_size, word_vocab_size + 2)  # pad, unk
        self.assertEqual(it.char_vocab_size, char_vocab_size + 2)  # pad, unk
        self.assertEqual(it.label_size, label_size + 1)            # pad

    def test_vocab_size_lower_off(self):
        word_vocab_size = 5
        char_vocab_size = 4
        label_size = 3

        # lower is not effective.
        it = IndexTransformer(lower=False)
        it.fit(self.x, self.y)
        self.assertEqual(it.word_vocab_size, word_vocab_size + 2)  # pad, unk
        self.assertEqual(it.char_vocab_size, char_vocab_size + 2)  # pad, unk
        self.assertEqual(it.label_size, label_size + 1)            # pad

    def test_vocab_size_with_initial_vocab(self):
        vocab = {'aaa', 'aab', 'aac'}
        word_vocab_size = 4 + len(vocab)
        char_vocab_size = 4
        label_size = 3

        # Add initial vocab.
        it = IndexTransformer(lower=True, initial_vocab=vocab)
        it.fit(self.x, self.y)
        self.assertEqual(it.word_vocab_size, word_vocab_size + 2)  # pad, unk
        self.assertEqual(it.char_vocab_size, char_vocab_size + 2)  # pad, unk
        self.assertEqual(it.label_size, label_size + 1)            # pad

    def test_transform_without_character(self):
        # No character feature.
        it = IndexTransformer(use_char=False)
        x, y = it.fit_transform(self.x, self.y)
        x, length = x

        # Check sequence length.
        self.assertEqual(len(x), len(self.x))
        self.assertEqual(len(y), len(self.y))

        # Check sequence type.
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_transform_with_character(self):
        # With character feature.
        it = IndexTransformer(use_char=True)
        X, y = it.fit_transform(self.x, self.y)
        words, chars, length = X

        # Check sequence length.
        self.assertEqual(len(words), len(self.x))
        self.assertEqual(len(chars), len(self.x))
        self.assertEqual(len(y), len(self.y))

        # Check sequence type.
        self.assertIsInstance(words, np.ndarray)
        self.assertIsInstance(chars, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_transform_unknown_token(self):
        it = IndexTransformer()
        it.fit(self.x, self.y)

        x_train, y_train = [['aaa']], [['X']]
        X, y = it.transform(x_train, y_train)
        words, chars, length = X

        # Check sequence length.
        self.assertEqual(len(words), len(x_train))
        self.assertEqual(len(chars), len(x_train))
        self.assertEqual(len(y), len(y_train))

        # Check sequence type.
        self.assertIsInstance(words, np.ndarray)
        self.assertIsInstance(chars, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_inverse_transform(self):
        it = IndexTransformer()
        x, y = it.fit_transform(self.x, self.y)
        _, _, length = x
        inv_y = it.inverse_transform(y, length)
        self.assertEqual(inv_y, self.y)

    def test_inverse_transform_unknown_token(self):
        x_train, y_train = [['a', 'b']], [['X', 'O']]
        it = IndexTransformer()
        it.fit(self.x, self.y)
        _, y = it.transform(x_train, y_train)
        inv_y = it.inverse_transform(y)
        self.assertNotEqual(inv_y, self.y)

    def test_inverse_transform_one_cat(self):
        x_train, y_train = [['a']], [['O']]
        it = IndexTransformer()
        it.fit(self.x, self.y)
        _, y = it.transform(x_train, y_train)
        inv_y = it.inverse_transform(y)
        self.assertNotEqual(inv_y, self.y)

    def test_save_and_load(self):
        it = IndexTransformer(lower=False)
        x1, y1 = it.fit_transform(self.x, self.y)
        x1_word, x1_char, x1_length = x1

        self.assertFalse(os.path.exists(self.preprocessor_file))
        it.save(self.preprocessor_file)
        self.assertTrue(os.path.exists(self.preprocessor_file))

        it = IndexTransformer.load(self.preprocessor_file)
        x2, y2 = it.transform(self.x, self.y)
        x2_word, x2_char, x2_length = x2

        np.testing.assert_array_equal(x1_word, x2_word)
        np.testing.assert_array_equal(x1_char, x2_char)
        np.testing.assert_array_equal(y1, y2)


class TestPadding(unittest.TestCase):

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
