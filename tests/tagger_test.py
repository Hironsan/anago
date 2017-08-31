import os
import unittest

import anago
from anago.config import ModelConfig
from anago.data.preprocess import WordPreprocessor


class TaggerTest(unittest.TestCase):

    def setUp(self):
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), '../models')

        model_config = ModelConfig()

        p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        model_config.vocab_size = len(p.vocab_word)
        model_config.char_vocab_size = len(p.vocab_char)

        weights = 'model_weights.h5'

        self.tagger = anago.Tagger(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)
        self.sent = 'President Obama is speaking at the White House.'

    def test_tagging(self):
        res = self.tagger.tag(self.sent)

        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], str)
        self.assertIsInstance(res[0][1], str)

        tag_set = {'O', 'LOC', 'PER', 'ORG', 'MISC'}
        for _, tag in res:
            self.assertIn(tag, tag_set)

    def test_get_entities(self):
        res = self.tagger.get_entities(self.sent)
        print(res)
        self.assertIsInstance(list(res.keys())[0], str)
        self.assertIsInstance(list(res.values())[0], list)
        self.assertIsInstance(list(res.values())[0][0], str)
