import unittest

import numpy as np

from anago.models.keras_model import SeqLabeling
from anago.config import Config


class ModelTest(unittest.TestCase):

    def test_build(self):
        config = Config()
        vocab = 10000
        config.char_vocab_size = 80
        embeddings = np.zeros((vocab, config.word_dim))
        model = SeqLabeling(config, embeddings, ntags=10)

    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_predict(self):
        config = Config()
        vocab = 10000
        config.char_vocab_size = 80
        embeddings = np.zeros((vocab, config.word_dim))
        model = SeqLabeling(config, embeddings, ntags=10)
        from keras.optimizers import Adam
        model.compile(loss=model.loss,
                      optimizer=Adam(lr=config.learning_rate)
                      )
        print(model.model.predict)
