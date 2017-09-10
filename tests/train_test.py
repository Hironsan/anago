import os
import unittest

import numpy as np

import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.data.preprocess import prepare_preprocessor
from anago.config import ModelConfig, TrainingConfig


class TrainerTest(unittest.TestCase):

    def test_train(self):
        DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
        SAVE_ROOT = os.path.join(os.path.dirname(__file__), '../models')  # trained model
        LOG_ROOT = os.path.join(os.path.dirname(__file__), '../logs')     # checkpoint, tensorboard
        embedding_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.100d.txt')

        model_config = ModelConfig()
        training_config = TrainingConfig()

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        test_path = os.path.join(DATA_ROOT, 'test.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)
        x_test, y_test = load_data_and_labels(test_path)

        p = prepare_preprocessor(np.r_[x_train, x_valid, x_test], y_train)  # np.r_ is for vocabulary expansion.
        p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
        model_config.char_vocab_size = len(p.vocab_char)

        trainer = anago.Trainer(model_config, training_config, checkpoint_path=LOG_ROOT, save_path=SAVE_ROOT,
                                preprocessor=p, embeddings=embeddings)
        trainer.train(x_train, y_train, x_test, y_test)
