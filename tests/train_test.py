import os
import unittest

from anago.reader import load_data_and_labels, load_word_embeddings

import anago
from anago.config import ModelConfig, TrainingConfig
from anago.models import SeqLabeling
from anago.preprocess import prepare_preprocessor

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

    def test_train(self):
        model_config = ModelConfig()
        training_config = TrainingConfig()

        train_path = os.path.join(DATA_ROOT, 'train.txt')
        valid_path = os.path.join(DATA_ROOT, 'valid.txt')
        x_train, y_train = load_data_and_labels(train_path)
        x_valid, y_valid = load_data_and_labels(valid_path)

        p = prepare_preprocessor(x_train, y_train)
        p.save(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
        embeddings = load_word_embeddings(p.vocab_word, EMBEDDING_PATH, model_config.word_embedding_size)
        model_config.char_vocab_size = len(p.vocab_char)

        model = SeqLabeling(model_config, embeddings, len(p.vocab_tag))

        trainer = anago.Trainer(model,
                                training_config,
                                checkpoint_path=LOG_ROOT,
                                save_path=SAVE_ROOT,
                                preprocessor=p,
                                embeddings=embeddings)
        trainer.train(x_train, y_train, x_valid, y_valid)

        model.save(os.path.join(SAVE_ROOT, 'model_weights.h5'))