import os
import unittest
from unittest.mock import MagicMock

from anago.data import conll
from anago.data.preprocess import get_vocabs, get_char_vocab, build_vocab, load_vocab, load_word_embeddings


class ProprocessTest(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en')
        self.datasets = conll.read_data_sets(data_path)

    def test_get_vocabs(self):
        words, tags = get_vocabs(self.datasets)
        true_tags = {'O', 'I-PER', 'B-ORG', 'B-LOC', 'B-PER', 'I-LOC', 'I-MISC', 'B-MISC', 'I-ORG'}
        self.assertEqual(tags, true_tags)
        self.assertIsInstance(words, set)

    def test_get_char_vocab(self):
        chars = get_char_vocab(self.datasets.train)
        self.assertIsInstance(chars, set)

    def test_build_vocab(self):
        config = MagicMock()
        config.save_path = os.path.join(os.path.dirname(__file__), 'data/')
        config.glove_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.50d.txt')
        vocab = build_vocab(self.datasets, config)
        loaded_vocab = load_vocab(config.save_path)
        self.assertEqual(vocab, loaded_vocab)

    def test_load_word_embeddings(self):
        config = MagicMock()
        config.save_path = os.path.join(os.path.dirname(__file__), 'data/')
        config.glove_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.50d.txt')
        config.dim = 50
        vocab = build_vocab(self.datasets, config)
        embeddings = load_word_embeddings(vocab.word, config.glove_path, config.dim)
        self.assertEqual(embeddings.shape[1], config.dim)
