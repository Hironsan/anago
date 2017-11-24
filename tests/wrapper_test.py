import os
import unittest
from pprint import pprint

import numpy as np

import anago
from anago.reader import load_data_and_labels, load_glove
from anago.utils import download

get_path = lambda path: os.path.join(os.path.dirname(__file__), path)
DATA_ROOT = get_path('../data/conll2003/en/ner')
SAVE_ROOT = get_path('models')  # trained model
LOG_ROOT = get_path('logs')     # checkpoint, tensorboard
EMBEDDING_PATH = get_path('../data/glove.6B/glove.6B.100d.txt')


class TrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        test_path = os.path.join(DATA_ROOT, 'test.txt')

        cls.x_train, cls.y_train = load_data_and_labels(train_path)
        cls.x_valid, cls.y_valid = load_data_and_labels(valid_path)
        cls.x_test, cls.y_test = load_data_and_labels(test_path)

        cls.embeddings = load_glove(EMBEDDING_PATH)

        cls.words = 'President Obama is speaking at the White House.'.split()

        cls.dir_path = 'models'

    def test_train(self):
        # Not use pre-trained word embeddings
        model = anago.Sequence(max_epoch=1)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)

        # Use pre-trained word embeddings
        model = anago.Sequence(max_epoch=1, embeddings=self.embeddings)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def test_eval(self):
        model = anago.Sequence(max_epoch=1, embeddings=self.embeddings)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)
        model.eval(self.x_test, self.y_test)

    def test_analyze(self):
        model = anago.Sequence(max_epoch=1, embeddings=self.embeddings)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)
        res = model.analyze(self.words)
        pprint(res)

        self.assertIn('words', res)
        self.assertIn('entities', res)

    def test_save(self):
        model = anago.Sequence(max_epoch=1, embeddings=self.embeddings)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)
        model.save(dir_path=self.dir_path)

        config_file = os.path.join(self.dir_path, model.config_file)
        weight_file = os.path.join(self.dir_path, model.weight_file)
        preprocessor_file = os.path.join(self.dir_path, model.preprocessor_file)

        self.assertTrue(os.path.exists(config_file))
        self.assertTrue(os.path.exists(weight_file))
        self.assertTrue(os.path.exists(preprocessor_file))

    def test_load(self):
        model = anago.Sequence(max_epoch=1, embeddings=self.embeddings)
        model.train(self.x_train, self.y_train, self.x_valid, self.y_valid)
        model.eval(self.x_test, self.y_test)
        model.save(dir_path=self.dir_path)

        model = anago.Sequence.load(self.dir_path)
        model.eval(self.x_test, self.y_test)

    def test_train_vocab_init(self):
        vocab = set()
        for words in np.r_[self.x_train, self.x_valid, self.x_test]:
            for word in words:
                vocab.add(word)
        model = anago.Sequence(max_epoch=15, embeddings=self.embeddings, log_dir='logs')
        model.train(self.x_train, self.y_train, self.x_test, self.y_test, vocab_init=vocab)
        model.save(dir_path=self.dir_path)

    def test_train_all(self):
        x_train = np.r_[self.x_train, self.x_valid, self.x_test]
        y_train = np.r_[self.y_train, self.y_valid, self.y_test]
        model = anago.Sequence(max_epoch=15, embeddings=self.embeddings, log_dir='logs')
        model.train(x_train, y_train, self.x_test, self.y_test)
        model.save(dir_path=self.dir_path)

    def test_download(self):
        dir_path = 'test_dir'
        url = 'https://storage.googleapis.com/chakki/datasets/public/models.zip'
        download(url, dir_path)
        model = anago.Sequence.load(dir_path)
        model.eval(self.x_test, self.y_test)
