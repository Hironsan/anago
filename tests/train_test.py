import os
import unittest

import anago
from anago.data.conll import extract_data


class TrainTest(unittest.TestCase):

    def test_train(self):
        filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/train.txt')
        sents, labels = extract_data(filename)

        hyperparams = None
        trainer = anago.Trainer(hyperparams)
        trainer.train(sents, labels)
