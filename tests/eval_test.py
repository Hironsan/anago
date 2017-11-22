import os
import unittest

from anago.metrics import get_entities, f1_score, F1score
from anago.reader import load_data_and_labels

import anago
from anago.config import ModelConfig
from anago.models import SeqLabeling
from anago.preprocess import WordPreprocessor

DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'models')


class EvaluatorTest(unittest.TestCase):

    def test_eval(self):
        test_path = os.path.join(DATA_ROOT, 'test.txt')
        x_test, y_test = load_data_and_labels(test_path)

        p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        config = ModelConfig()
        config.vocab_size = len(p.vocab_word)
        config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(config, ntags=len(p.vocab_tag))
        model.load(filepath=os.path.join(SAVE_ROOT, 'model_weights.h5'))

        evaluator = anago.Evaluator(model, preprocessor=p)
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
