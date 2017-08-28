import os
import unittest

import anago
from anago.data.reader import load_data_and_labels
from anago.data.metrics import get_entities, f1_score, F1score
from anago.config import Config


class EvaluatorTest(unittest.TestCase):

    def test_eval(self):
        config = Config()
        weights_file = 'model_weights_01_0.03.h5'

        test_path = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner/test.txt')
        x_test, y_test = load_data_and_labels(test_path)

        evaluator = anago.Evaluator(config, weights_file)
        evaluator.eval(x_test, y_test)


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
        f1 = f1_score(y_true, y_pred, seq_len)
        recall = 1.0 / 2
        precision = 1.0
        f_true = 2 * recall * precision / (recall + precision)
        self.assertEqual(f1, f_true)


class F1scoreTest(unittest.TestCase):

    def test_calc_f1score(self):
        f1score = F1score()

    def test_count_correct_and_pred(self):
        f1score = F1score()
