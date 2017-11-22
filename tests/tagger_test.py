import os
import unittest
from pprint import pprint

import anago
from anago.config import ModelConfig
from anago.models import SeqLabeling
from anago.preprocess import WordPreprocessor

SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')


class TaggerTest(unittest.TestCase):

    def setUp(self):
        p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))

        config = ModelConfig()
        config.vocab_size = len(p.vocab_word)
        config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(config, ntags=len(p.vocab_tag))
        model.load(filepath=os.path.join(SAVE_ROOT, 'model_weights.h5'))

        self.tagger = anago.Tagger(model, preprocessor=p)
        self.sent = 'President Obama is speaking at the White House.'

    def test_analyze(self):
        res = self.tagger.analyze(self.sent)
        pprint(res)

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
