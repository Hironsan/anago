import os
import unittest

from anago.utils import load_data_and_labels
from anago.models import BiLSTMCRF
from anago.preprocess import StaticPreprocessor, DynamicPreprocessor
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

    def setUp(self):
        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        self.x_train, self.y_train = load_data_and_labels(train_path)
        self.x_valid, self.y_valid = load_data_and_labels(valid_path)

    def test_train(self):
        p = StaticPreprocessor()
        p.fit(self.x_train, self.y_train)
        x_train, y_train = p.transform(self.x_train, self.y_train)
        x_valid, y_valid = p.transform(self.x_valid, self.y_valid)

        model = BiLSTMCRF(char_vocab_size=len(p.char_dic),
                          word_vocab_size=len(p.word_dic),
                          ntags=len(p.label_dic))
        model.build_model()
        dp = DynamicPreprocessor(n_labels=len(p.label_dic))

        trainer = Trainer(model, model.get_loss(), preprocessor=dp)
        trainer.train(x_train, y_train, x_valid, y_valid)

    def test_predict(self):
        pass

    def test_save(self):
        pass

    def test_load(self):
        pass
