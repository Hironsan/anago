import os
import unittest

from anago.data import reader

class TestReader(unittest.TestCase):

    def test_read(self):
        data_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        train, valid, test, word_to_id, entity_to_id = reader.conll_raw_data(data_path)
        print(test['X'][0])
        print(word_to_id)
        print(entity_to_id)
