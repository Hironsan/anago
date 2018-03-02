import os
import unittest

from anago.reader import load_data_and_labels, load_glove_vocab, load_word_embeddings, batch_iter

from anago.preprocess import prepare_preprocessor


class ReaderTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_extract(self):
        sents, labels = load_data_and_labels(self.filename)
        self.assertTrue(len(sents) == len(labels))

    def test_load_glove_vocab(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
        filename = os.path.join(self.DATA_DIR, 'glove.50d.txt')
        vocab = load_glove_vocab(filename)
        true_vocab = {'the', ',', '.'}
        self.assertEqual(vocab, true_vocab)

    def test_load_word_embeddings(self):
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
        filename = os.path.join(self.DATA_DIR, 'glove.50d.txt')
        vocab = load_glove_vocab(filename)
        vocab = {w: i for i, w in enumerate(vocab)}
        dim = 50
        embeddings = load_word_embeddings(vocab, filename, dim=dim)
        self.assertEqual(embeddings.shape[1], dim)

        dim = 10
        embeddings = load_word_embeddings(vocab, filename, dim=dim)
        self.assertEqual(embeddings.shape[1], dim)

        dim = 1000
        actual_dim = 50
        embeddings = load_word_embeddings(vocab, filename, dim=dim)
        self.assertNotEqual(embeddings.shape[1], dim)
        self.assertEqual(embeddings.shape[1], actual_dim)

    def test_batch_iter(self):
        sents, labels = load_data_and_labels(self.filename)
        batch_size = 32
        p = prepare_preprocessor(sents, labels)
        steps, batches = batch_iter(list(zip(sents, labels)), batch_size, preprocessor=p)
        self.assertEqual(len([_ for _ in batches]), steps)  # Todo: infinite loop
