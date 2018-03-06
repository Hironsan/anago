import os
import unittest
from pprint import pprint

import anago
from anago.models import BiLSTMCRF
from anago.preprocess import StaticPreprocessor, DynamicPreprocessor

DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')


class TaggerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')

        # Load preprocessor
        p = StaticPreprocessor.load(preprocessor_file)
        dp = DynamicPreprocessor(len(p.label_dic))

        # Load the model.
        model = BiLSTMCRF.load(weights_file, params_file)

        # Build a tagger
        cls.tagger = anago.Tagger(model, preprocessor=p, dynamic_preprocessor=dp)

        cls.sent = 'President Obama is speaking at the White House.'

    def test_predict(self):
        res = self.tagger.predict(self.sent)
        pprint(res)

    def test_analyze(self):
        res = self.tagger.analyze(self.sent)
        pprint(res)
