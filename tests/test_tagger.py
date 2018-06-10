import os
import unittest

import numpy as np
import tensorflow as tf

import anago
from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer

DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')


class TestTagger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')

        # Load preprocessor
        p = IndexTransformer.load(preprocessor_file)

        # Load the model.
        model = BiLSTMCRF.load(weights_file, params_file)

        # Build a tagger
        cls.tagger = anago.Tagger(model, preprocessor=p)

        cls.sent = 'President Obama is speaking at the White House.'

    def test_predict_proba(self):
        res = self.tagger.predict_proba(self.sent)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), len(self.sent.split()))

        res = self.tagger.predict_proba('Obama')
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), len('Obama'.split()))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            res = self.tagger.predict_proba('')

    def test_analyze(self):
        res = self.tagger.analyze(self.sent)
        self.assertIsInstance(res, dict)
        self.assertIn('words', res)
        self.assertIn('entities', res)
        self.assertIsInstance(res['words'], list)
        self.assertIsInstance(res['entities'], list)
        for w in res['words']:
            self.assertIsInstance(w, str)
        for e in res['entities']:
            self.assertIsInstance(e, dict)
            self.assertIn('beginOffset', e)
            self.assertIn('endOffset', e)
            self.assertIn('score', e)
            self.assertIn('text', e)
            self.assertIn('type', e)

    def test_predict_labels(self):
        res = self.tagger.predict(self.sent)
        self.assertEqual(len(res), len(self.sent.split()))
        self.assertIsInstance(res, list)
        for tag in res:
            self.assertIsInstance(tag, str)
