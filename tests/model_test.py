import os
import unittest

import numpy as np
from keras.optimizers import Adam

from anago.data.reader import load_data_and_labels, batch_iter
from anago.data.preprocess import prepare_preprocessor
from anago.models.keras_model import SeqLabeling
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

        train_steps, train_batches = batch_iter(
            list(zip(X, y)), self.config.batch_size, preprocessor=p)

        valid_steps, valid_batches = batch_iter(
            list(zip(X, y)), self.config.batch_size, preprocessor=p)


        model = SeqLabeling(self.config, self.embeddings, ntags=len(p.vocab_tag))
        model.compile(loss=model.loss,
                      optimizer=Adam(lr=self.config.learning_rate)
                      )

        model.model.fit_generator(train_batches, train_steps, epochs=15,
                                  validation_data=valid_batches, validation_steps=valid_steps)

        X, y = p(X, y)
        y_true = np.argmax(y, -1)
        seq_length = np.argmin(y_true, -1)
        y_pred = model.predict(X, seq_length)
        print(y_true)
        print(type(y_true))
        print(y_pred)
        print(type(y_pred))

    def test_save(self):
        pass

    def test_load(self):
        pass
