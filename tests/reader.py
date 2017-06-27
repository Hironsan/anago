import os
import unittest

from anago.data.reader import reader, Reader


class TestReader(unittest.TestCase):

    def setUp(self):
        self.dir_path = os.path.join(os.path.dirname(__file__), '../data/raw/KWDLC-1.0')

    def test_traverse(self):
        r = Reader()
        r.read_entity(self.dir_path)

    """
    def test_reader(self):
        dir_path = os.path.join(os.path.dirname(__file__), '../data/raw/KWDLC-1.0/dat/rel/')
        print(dir_path)
        docs = reader(dir_path)
        print(docs[0])
    """