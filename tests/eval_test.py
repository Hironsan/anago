import os
import unittest

import anago
from anago.data.reader import DataSet, load_data_and_labels
from anago.data.metrics import get_entities, f1_score


class TrainTest(unittest.TestCase):

    def test_train(self):
        filename = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/train.txt')
        sents, labels = load_data_and_labels(filename)
        dataset = DataSet(sents, labels)

        hyperparams = None
        evaluator = anago.Evaluator(hyperparams)
        evaluator.eval(dataset)


class EvalTest(unittest.TestCase):

    def test_get_entities(self):
        seq = ['B-PERSON', 'I-PERSON', 'O', 'B-LOC', 'I-LOC']
        chunks = get_entities(seq)
        expected_chunks = [('PERSON', 0, 2), ('LOC', 3, 5)]
        self.assertEqual(chunks, expected_chunks)

        seq = ['B-PERSON', 'I-PERSON', 'O', 'B-LOC', 'O']
        chunks = get_entities(seq)
        expected_chunks = [('PERSON', 0, 2), ('LOC', 3, 4)]
        self.assertEqual(chunks, expected_chunks)

        seq = ['B-PERSON', 'I-PERSON', 'O', 'I-LOC', 'O']
        chunks = get_entities(seq)
        expected_chunks = [('PERSON', 0, 2)]
        self.assertEqual(chunks, expected_chunks)

        seq = ['B-PERSON', 'I-PERSON', 'O', 'O', 'B-LOC']
        chunks = get_entities(seq)
        expected_chunks = [('PERSON', 0, 2), ('LOC', 4, 5)]
        self.assertEqual(chunks, expected_chunks)

        seq = ['O', 'B-PERSON', 'O', 'O', 'B-LOC']
        chunks = get_entities(seq)
        expected_chunks = [('PERSON', 1, 2), ('LOC', 4, 5)]
        self.assertEqual(chunks, expected_chunks)

    def test_run_evaluate(self):
        y_true = [['B-PERSON', 'I-PERSON', 'O', 'O', 'B-LOC']]
        y_pred = [['B-PERSON', 'I-PERSON', 'O', 'O', 'I-ORG']]
        seq_len = [5]
        acc, f1 = f1_score(y_true, y_pred, seq_len)
        self.assertEqual(acc, 0.8)
        recall = 1.0 / 2
        precision = 1.0
        f_true = 2 * recall * precision / (recall + precision)
        self.assertEqual(f1, f_true)
