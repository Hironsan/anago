import unittest

from anago.data_utils import *


class DataUtilsTest(unittest.TestCase):

    def setUp(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

    def test_conll_dataset(self):
        filename = os.path.join(self.DATA_DIR, 'conll.txt')
        dataset = CoNLLDataset(filename)
        for i in dataset:
            print(i)

    def test_get_vocabs(self):
        pass

    def test_get_char_vocabs(self):
        pass

    def test_load_glove_vocab(self):
        filename = os.path.join(self.DATA_DIR, 'glove.50d.txt')
        vocab = load_glove_vocab(filename)
        true_vocab = {'the', ',', '.'}
        self.assertEqual(vocab, true_vocab)

    def test_write_vocab(self):
        vocab = {'cat': 0, 'dog': 1, 'pig': 2}
        filename = os.path.join(self.DATA_DIR, 'tmp.txt')
        write_vocab(vocab, filename)
        loaded_vocab = load_vocab(filename)
        self.assertEqual(loaded_vocab, vocab)
        os.remove(filename)

    def test_load_vocab(self):
        words_file = os.path.join(self.DATA_DIR, 'words.txt')
        chars_file = os.path.join(self.DATA_DIR, 'chars.txt')
        tags_file  = os.path.join(self.DATA_DIR, 'tags.txt')

        words = load_vocab(words_file)
        chars = load_vocab(chars_file)
        tags = load_vocab(tags_file)

        f = lambda file: {l.strip():i for i, l in enumerate(open(file))}
        true_words = f(words_file)
        true_chars = f(chars_file)
        true_tags  = f(tags_file)

        self.assertEqual(words, true_words)
        self.assertEqual(chars, true_chars)
        self.assertEqual(tags, true_tags)

    def export_trimmed_glove_vectors(self):
        vocab = {'cat': 0, 'dog': 1, 'pig': 2}
        glove_file = os.path.join(self.DATA_DIR, 'glove.50d.txt')
        trimmed_file = os.path.join(self.DATA_DIR, 'trimmed.txt')
        dim = 50
        export_trimmed_glove_vectors(vocab, glove_file, trimmed_file, dim)
        self.assertTrue(os.path.exists(trimmed_file))
        os.remove(trimmed_file)
