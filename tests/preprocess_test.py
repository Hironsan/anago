import os
import unittest

from sklearn.externals import joblib

from anago.data import reader
from anago.data.preprocess import WordPreprocessor, UNK


class WordPreprocessorTest(unittest.TestCase):

    def test_preprocessor(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = reader.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels

        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X, y = p.transform(X, y)
        chars, words = X[0]
        char, word = chars[0][0], words[0]
        tag = y[0][0]
        self.assertIsInstance(word, int)
        self.assertIsInstance(char, int)
        self.assertIsInstance(tag, int)
        self.assertIsInstance(p.inverse_transform(y[0])[0], str)

    def test_unknown_word(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = reader.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels
        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X = [['$unknownword$', 'あ']]
        y = [['O', 'O']]
        X, y = p.transform(X, y)

    def test_vocab_init(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = reader.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels
        unknown_word = 'unknownword'
        X_test, y_test = [[unknown_word]], [['O']]

        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0][1]
        self.assertEqual(words, [p.vocab_word[UNK]])

        vocab_init = {unknown_word}
        preprocessor = WordPreprocessor(vocab_init=vocab_init)
        p = preprocessor.fit(X, y)
        X_pred, _ = p.transform(X_test, y_test)
        words = X_pred[0][1]
        self.assertNotEqual(words, [p.vocab_word[UNK]])

    def test_save(self):
        preprocessor = WordPreprocessor()
        filename = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        joblib.dump(preprocessor, filename)
        self.assertTrue(os.path.exists(filename))
        if os.path.exists(filename):
            os.remove(filename)

    def test_load(self):
        train_dir = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/')
        datasets = reader.read_data_sets(train_dir)
        X, y = datasets.train.sents, datasets.train.labels

        preprocessor = WordPreprocessor()
        p = preprocessor.fit(X, y)
        filename = os.path.join(os.path.dirname(__file__), 'data/preprocessor.pkl')
        joblib.dump(p, filename)
        self.assertTrue(os.path.exists(filename))
        loaded_p = joblib.load(filename)

        self.assertEqual(loaded_p.transform(X, y), p.transform(X, y))

        if os.path.exists(filename):
            os.remove(filename)
