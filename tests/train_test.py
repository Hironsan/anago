import os
import unittest

import anago
from anago.data.reader import load_data_and_labels
from anago.config import Config


class TrainerTest(unittest.TestCase):

    def test_train(self):
        config = Config()
        config.data_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en')
        config.save_path = os.path.join(os.path.dirname(__file__), '../garbage/')
        config.log_dir = os.path.join(os.path.dirname(__file__), '../logs/')
        config.glove_path = os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.300d.txt')

        train_path = os.path.join(config.data_path, 'train.txt')
        valid_path = os.path.join(config.data_path, 'valid.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)

        trainer = anago.Trainer(config)
        trainer.train(x_valid, y_valid, x_valid, y_valid)
        trainer.train(x_train, y_train, x_valid, y_valid)
