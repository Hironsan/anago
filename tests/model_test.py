import os
import unittest

import numpy as np
from anago.reader import load_data_and_labels
from keras.optimizers import Adam

from anago.config import ModelConfig, TrainingConfig
from anago.models import SeqLabeling
from anago.preprocess import prepare_preprocessor


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        vocab = 10000
        self.model_config.char_vocab_size = 80
        self.embeddings = np.zeros((vocab, self.model_config.word_embedding_size))
        self.filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')
        self.valid_file = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/valid.txt')

    def test_build(self):
        model = SeqLabeling(self.model_config, self.embeddings, ntags=10)

    def test_compile(self):
        model = SeqLabeling(self.model_config, self.embeddings, ntags=10)
        model.compile(loss=model.crf.loss,
                      optimizer=Adam(lr=self.training_config.learning_rate)
                      )

    def test_predict(self):
        X, y = load_data_and_labels(self.filename)
        X, y = X[:100], y[:100]
        p = prepare_preprocessor(X, y)
        self.model_config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(self.model_config, self.embeddings, ntags=len(p.vocab_tag))
        model.predict(p.transform(X))

    def test_save(self):
        pass

    def test_load(self):
        pass
