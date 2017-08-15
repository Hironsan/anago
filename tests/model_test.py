import unittest

import numpy as np

from anago.models.keras_model import LSTMCrf
from anago.config import Config


class ModelTest(unittest.TestCase):

    def test_build(self):
        config = Config()
        vocab = 10000
        config.char_vocab_size = 80
        embeddings = np.zeros((vocab, config.word_dim))
        model = LSTMCrf(config, embeddings, ntags=10)
        model.build()

    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_predict(self):
        pass
