import os

from keras.optimizers import Adam

from anago.data.metrics import get_callbacks
from anago.data.preprocess import prepare_preprocessor
from anago.data.reader import load_word_embeddings, batch_iter
from anago.models import SeqLabeling


class Trainer(object):

    def __init__(self, config):
        self.config = config

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        p = prepare_preprocessor(x_train, y_train)
        embeddings = load_word_embeddings(p.vocab_word, self.config.glove_path, self.config.word_dim)
        self.config.char_vocab_size = len(p.vocab_char)

        train_steps, train_batches = batch_iter(
            list(zip(x_train, y_train)), self.config.batch_size, preprocessor=p)
        valid_steps, valid_batches = batch_iter(
            list(zip(x_valid, y_valid)), self.config.batch_size, preprocessor=p)

        model = SeqLabeling(self.config, embeddings, len(p.vocab_tag))
        model.compile(loss=model.crf.loss,
                      optimizer=Adam(lr=self.config.learning_rate),
                      )
        callbacks = get_callbacks(log_dir=self.config.log_dir,
                                  save_dir=self.config.save_path,
                                  valid=(valid_steps, valid_batches, p))
        model.fit_generator(generator=train_batches,
                            steps_per_epoch=train_steps,
                            epochs=self.config.max_epoch,
                            callbacks=callbacks)
        p.save(os.path.join(self.config.save_path, 'preprocessor.pkl'))
