import os

import numpy as np

from anago.models import BiLSTMCRF
from anago.preprocessing import IndexTransformer, DynamicPreprocessor
from anago.tagger import Tagger
from anago.trainer import Trainer


class Sequence(object):

    config_file = 'config.json'
    weight_file = 'model_weights.h5'
    preprocessor_file = 'preprocessor.pkl'

    def __init__(self, char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                 word_lstm_units=100, dropout=0.5, char_feature=True, crf=True,
                 batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=15, early_stopping=True, patience=3,
                 train_embeddings=True, max_checkpoints_to_keep=5, log_dir=None,
                 embeddings=()):

        self.model = None
        self.p = None
        self.dp = None
        self.log_dir = log_dir
        self.embeddings = embeddings

    def train(self, x_train, y_train, x_valid=None, y_valid=None, initial_vocab=None):
        self.p = IndexTransformer(initial_vocab=initial_vocab)
        self.p.fit(x_train, y_train)
        x_train, y_train = self.p.transform(x_train, y_train)
        x_valid, y_valid = self.p.transform(x_valid, y_valid)
        self.dp = DynamicPreprocessor(num_labels=self.p.label_size)

        # Build a model.
        self.model = BiLSTMCRF(char_vocab_size=self.p.char_vocab_size,
                               word_vocab_size=self.p.word_vocab_size,
                               num_labels=self.p.label_size)
        self.model.build()

        # Train the model.
        trainer = Trainer(self.model, self.model.get_loss(), preprocessor=self.dp,
                          inverse_transform=self.p.inverse_transform)
        trainer.train(x_train, y_train, x_valid, y_valid)

    def eval(self, x_test, y_test):
        if self.model:
            evaluator = Evaluator(self.model, preprocessor=self.p)
            evaluator.eval(x_test, y_test)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def analyze(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze(words)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def save(self, dir_path):
        self.p.save(os.path.join(dir_path, self.preprocessor_file))
        self.model_config.save(os.path.join(dir_path, self.config_file))
        self.model.save(os.path.join(dir_path, self.weight_file))

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()

        # Load preprocessor
        self.p = IndexTransformer.load(preprocessor_file)
        self.dp = DynamicPreprocessor(p.label_size)

        # Load the model.
        self.model = BiLSTMCRF.load(weights_file, params_file)

        return self
