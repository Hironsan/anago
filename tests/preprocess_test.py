import os
import unittest
from unittest.mock import MagicMock

from anago.data import conll
from anago.data.preprocess import get_vocabs, get_char_vocab, build_vocab, load_vocab, load_word_embeddings
from anago.data.preprocess import WordPreprocessor, UNK


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


class WordPreprocessorTest(unittest.TestCase):

    def test_preprocessor(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = conll.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels

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
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = conll.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = p.transform(X, y)

    def test_vocab_init(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = conll.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels
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
