import os
import unittest

from anago.utils import load_data_and_labels
from anago.models import BiLSTMCRF
from anago.preprocess import IndexTransformer, DynamicPreprocessor
from anago.trainer import Trainer

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

        cls.weights_file = os.path.join(SAVE_ROOT, 'weights.h5')
        cls.params_file = os.path.join(SAVE_ROOT, 'params.json')
        cls.preprocessor_file = os.path.join(SAVE_ROOT, 'preprocessor.pickle')

    def setUp(self):
        # Load datasets.
        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)

        # Transform datasets.
        self.p = IndexTransformer()
        self.p.fit(x_train, y_train)
        self.x_train, self.y_train = self.p.transform(x_train, y_train)
        self.x_valid, self.y_valid = self.p.transform(x_valid, y_valid)
        self.dp = DynamicPreprocessor(num_labels=self.p.label_size)

        # Build a model.
        self.model = BiLSTMCRF(char_vocab_size=self.p.char_vocab_size,
                               word_vocab_size=self.p.word_vocab_size,
                               num_labels=self.p.label_size)
        self.model.build()

    def test_train(self):
        # Train the model.
        trainer = Trainer(self.model, self.model.get_loss(), preprocessor=self.dp,
                          inverse_transform=self.p.inverse_transform)
        trainer.train(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def test_train_without_crf(self):
        model = BiLSTMCRF(char_vocab_size=self.p.char_vocab_size,
                          word_vocab_size=self.p.word_vocab_size,
                          num_labels=self.p.label_size,
                          use_crf=False)
        model.build()
        trainer = Trainer(self.model, self.model.get_loss(), preprocessor=self.dp,
                          inverse_transform=self.p.inverse_transform)
        trainer.train(self.x_train, self.y_train, self.x_valid, self.y_valid)

    def test_save(self):
        # Train the model.
        trainer = Trainer(self.model, self.model.get_loss(), preprocessor=self.dp,
                          inverse_transform=self.p.inverse_transform, max_epoch=1)
        trainer.train(self.x_train, self.y_train, self.x_valid, self.y_valid)

        # Save the model.
        self.model.save(self.weights_file, self.params_file)
        self.p.save(self.preprocessor_file)
