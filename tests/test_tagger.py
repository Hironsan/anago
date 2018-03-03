import os
import unittest
from pprint import pprint

import anago
from anago.models import BiLSTMCRF
from anago.preprocess import StaticPreprocessor


class TaggerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        params = {}
        save_root = os.path.join(os.path.dirname(__file__), 'models')

        p = StaticPreprocessor.load(os.path.join(save_root, 'preprocessor.pkl'))
        model = BiLSTMCRF.load(os.path.join(save_root, 'model_weights.h5'), params)
        cls.tagger = anago.Tagger(model, preprocessor=p)
        cls.sent = 'President Obama is speaking at the White House.'

    def test_analyze(self):
        res = self.tagger.analyze(self.sent)
        pprint(res)
