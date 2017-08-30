import os
import unittest

import anago
from anago.data.reader import load_data_and_labels
from anago.config import Config


class TrainerTest(unittest.TestCase):

    def test_train(self):
        config = Config()
        train_path = os.path.join(config.data_path, 'train.txt')
        valid_path = os.path.join(config.data_path, 'valid.txt')
        test_path = os.path.join(config.data_path, 'test.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)
        x_test, y_test = load_data_and_labels(test_path)

        x_train, y_train = x_train[:100], y_train[:100]
        x_valid, y_valid = x_train[:100], y_train[:100]
        trainer = anago.Trainer(config)
        trainer.train(x_train, y_train, x_valid, y_valid)
