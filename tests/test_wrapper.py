import os
import unittest
from pprint import pprint

import numpy as np
from keras.callbacks import ModelCheckpoint

import anago
from anago.utils import load_data_and_labels, load_glove

get_path = lambda path: os.path.join(os.path.dirname(__file__), path)
DATA_ROOT = get_path('../data/conll2003/en/ner')
SAVE_ROOT = get_path('models')  # trained model
LOG_ROOT = get_path('logs')     # checkpoint, tensorboard
EMBEDDING_PATH = get_path('../data/glove.6B/glove.6B.100d.txt')


class TestWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        test_path = os.path.join(DATA_ROOT, 'test.txt')

        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)
        cls.x_test, cls.y_test = load_data_and_labels(test_path)
        cls.x_train = np.r_[x_train, x_valid]
        cls.y_train = np.r_[y_train, y_valid]

        cls.embeddings = load_glove(EMBEDDING_PATH)
        cls.text = 'President Obama is speaking at the White House.'
        cls.dir_path = 'models'

    def test_train_without_pretrained_embedding(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_train_with_pretrained_embedding(self):
        model = anago.Sequence(embeddings=self.embeddings)
        model.fit(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_score(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train)
        score = model.score(self.x_test, self.y_test)
        self.assertIsInstance(score, float)

    def test_analyze(self):
        model = anago.Sequence()
        model.fit(self.x_train, self.y_train)
        res = model.analyze(self.text)
        pprint(res)

        self.assertIn('words', res)
        self.assertIn('entities', res)

    def test_save_and_load(self):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')

        model = anago.Sequence()
        model.fit(self.x_train, self.y_train)
        model.save(weights_file, params_file, preprocessor_file)
        score1 = model.score(self.x_test, self.y_test)

        self.assertTrue(weights_file)
        self.assertTrue(params_file)
        self.assertTrue(preprocessor_file)

        model = anago.Sequence.load(weights_file, params_file, preprocessor_file)
        score2 = model.score(self.x_test, self.y_test)

        self.assertEqual(score1, score2)

    def test_train_vocab_init(self):
        vocab = set()
        for words in np.r_[self.x_train, self.x_test, self.x_test]:
            for word in words:
                vocab.add(word)
        model = anago.Sequence(initial_vocab=vocab, embeddings=self.embeddings)
        model.fit(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_load(self):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')
        model = anago.Sequence.load(weights_file, params_file, preprocessor_file)
        score = model.score(self.x_test, self.y_test)
        print(score)

    def test_train_callbacks(self):
        weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        params_file = os.path.join(SAVE_ROOT, 'params.json')
        preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')

        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.4f}']) + '.h5'
        callback = ModelCheckpoint(os.path.join(log_dir, file_name),
                                   monitor='f1',
                                   save_weights_only=True)
        vocab = set()
        for words in np.r_[self.x_train, self.x_test, self.x_test]:
            for word in words:
                vocab.add(word)
        model = anago.Sequence(initial_vocab=vocab, embeddings=self.embeddings)
        model.fit(self.x_train, self.y_train, self.x_test, self.y_test,
                  epochs=30, callbacks=[callback])
        model.save(weights_file, params_file, preprocessor_file)
