import os
import unittest

import anago
from anago.data.conll import DataSet, extract_data


class TrainTest(unittest.TestCase):

    def test_train(self):
        filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/train.txt')
        sents, labels = extract_data(filename)
        dataset = DataSet(sents, labels)

        hyperparams = None
        evaluator = anago.Evaluator(hyperparams)
        evaluator.eval(dataset)
