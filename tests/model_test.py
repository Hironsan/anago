import unittest

import numpy as np

from anago.models.keras_model import LSTMCrf
from anago.config import Config


class ModelTest(unittest.TestCase):

    def test_build_model(self):
        config = Config()
        vocab = 10000
        config.char_vocab_size = 80
        embeddings = np.zeros((vocab, config.word_dim))
        model = LSTMCrf(config, embeddings, ntags=10)
        model._build_model()
