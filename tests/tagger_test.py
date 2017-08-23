import unittest

import anago
from anago.config import Config


class TaggerTest(unittest.TestCase):

    def setUp(self):
        config = Config()
        weights_file = 'model_weights_02_0.09.h5'
        self.tagger = anago.Tagger(config, weights_file)
        self.sent = 'President Obama is speaking at the White House.'

    def test_tagging(self):
        res = self.tagger.tag(self.sent)

        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], str)
        self.assertIsInstance(res[0][1], str)

        tag_set = {'O', 'Location', 'Person', 'Organization', 'Misc'}
        for _, tag in res:
            self.assertIn(tag, tag_set)

    def test_get_entities(self):
        res = self.tagger.get_entities(self.sent)

        self.assertIsInstance(list(res.keys())[0], str)
        self.assertIsInstance(list(res.values())[0], list)
        self.assertIsInstance(list(res.values())[0][0], str)
