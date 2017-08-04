import os
import unittest

from anago.data.conll import *
from anago.data.preprocess import *


class TestConll(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/test.txt')

    def test_extract(self):
        sents, labels = extract_data(self.filename)
        self.assertTrue(len(sents) == len(labels))

    def test_dataset(self):
        sents, labels = extract_data(self.filename)
        d = DataSet(sents, labels)
        sents, labels = d.next_batch(batch_size=32)
        self.assertTrue(len(sents) == len(labels))

    def test_read_datasest(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = read_data_sets(train_dir)
        self.assertTrue(hasattr(datasets, 'train'))
        self.assertTrue(hasattr(datasets, 'valid'))
        self.assertTrue(hasattr(datasets, 'test'))

    def test_dataset_with_preprocessing(self):
        X, y = extract_data(self.filename)
        d = DataSet(X, y)
        sents, tags = d.next_batch(batch_size=1)
        word, tag = sents[0][0], tags[0][0]
        self.assertIsInstance(word, str)
        self.assertIsInstance(tag, str)

        p = WordPreprocessor()
        p = p.fit(X, y)
        d = DataSet(X, y, preprocessor=p)
        sents, tags = d.next_batch(batch_size=1)
        chars, words = sents[0]
        word, char, tag = words[0], chars[0][0], tags[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
