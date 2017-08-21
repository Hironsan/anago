import os
import unittest

import numpy as np
from keras.optimizers import Adam

from anago.data.reader import load_data_and_labels
from anago.data.preprocess import prepare_preprocessor
from anago.models.models import SeqLabeling
from anago.config import Config


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        vocab = 10000
        self.config.char_vocab_size = 80
        self.embeddings = np.zeros((vocab, self.config.word_dim))
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/test.txt')
        self.valid_file = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/valid.txt')

    def test_build(self):
        model = SeqLabeling(self.config, self.embeddings, ntags=10)

    def test_compile(self):
        model = SeqLabeling(self.config, self.embeddings, ntags=10)
        model.compile(model.loss, optimizer=Adam(lr=self.config.learning_rate))

    def test_predict(self):
        X, y = load_data_and_labels(self.filename)
        X, y = X[:100], y[:100]
        p = prepare_preprocessor(X, y)
        self.config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(self.config, self.embeddings, ntags=len(p.vocab_tag))
        model.compile(loss=model.loss,
                      optimizer=Adam(lr=self.config.learning_rate)
                      )
        model.predict(X, sequence_lengths)

    def test_save(self):
        pass

    def test_load(self):
        pass
