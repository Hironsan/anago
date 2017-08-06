import unittest

import anago


class TaggerTest(unittest.TestCase):

    def setUp(self):
        self.sent = 'President Obama is speaking at the White House.'

    def test_tagging(self):
        tagger = anago.Tagger()
        res = tagger.tag(self.sent)

        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], str)
        self.assertIsInstance(res[0][1], str)

    def test_get_entities(self):
        tagger = anago.Tagger()
        res = tagger.get_entities(self.sent)

        self.assertIsInstance(list(res.keys())[0], str)
        self.assertIsInstance(list(res.values())[0], list)
        self.assertIsInstance(list(res.values())[0][0], str)
