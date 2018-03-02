import os
import unittest

from anago.reader import load_data_and_labels, batch_iter

from anago.preprocess import DynamicPreprocessor, StaticPreprocessor


class ReaderTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')

    def test_extract(self):
        sents, labels = load_data_and_labels(self.filename)
        self.assertTrue(len(sents) == len(labels))

    def test_batch_iter(self):
        sents, labels = load_data_and_labels(self.filename)
        batch_size = 32
        p = DynamicPreprocessor()
        steps, batches = batch_iter(list(zip(sents, labels)), batch_size, preprocessor=p)
        self.assertEqual(len([_ for _ in batches]), steps)  # Todo: infinite loop
