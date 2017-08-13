from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

from anago.models.keras_model import LSTMCrf
from anago.data.preprocess import load_word_embeddings
from anago.data.conll import load_glove_vocab, WordPreprocessor, DataSet
from anago.data_utils import pad_sequences, get_chunks
from anago.general_utils import Progbar, get_logger


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(os.path.join(config.log_dir, 'log.txt'))
        self.sequence_lengths = 0

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        p = WordPreprocessor()
        p = p.fit(x_train, y_train)
        train = DataSet(x_train, y_train, preprocessor=p)
        embeddings = load_word_embeddings(p.vocab_word, self.config.glove_path, self.config.word_dim)
        self.config.char_vocab_size = len(p.vocab_char)

        model = LSTMCrf(self.config, embeddings, len(p.vocab_tag))
        model = model.build()
        model.compile(loss=self.loss,
                      optimizer=RMSprop(lr=self.config.learning_rate),
                      )##metrics=['acc'])

        for epoch in range(self.config.max_epoch):
            self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, self.config.max_epoch))

            nbatches = (len(train.sents) + self.config.batch_size - 1) // self.config.batch_size
            prog = Progbar(target=nbatches)
            for i in range(nbatches):
                words, labels = train.next_batch(self.config.batch_size)
                words, chars, labels = self.pad_sequence(words, labels, len(p.vocab_tag))
                train_loss = model.train_on_batch([words, chars], labels)
                prog.update(i + 1, [('train loss', train_loss)])

    def pad_sequence(self, words, labels, ntags):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = np.asarray([to_categorical(y, num_classes=ntags) for y in labels])

        if self.config.use_char:
            char_ids, word_ids = zip(*words)
            word_ids, self.sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            word_ids, char_ids = np.asarray(word_ids), np.asarray(char_ids)
            return word_ids, char_ids, labels
        else:
            word_ids, self.sequence_lengths = pad_sequences(words, 0)
            word_ids = np.asarray(word_ids)
            return np.asarray(word_ids), labels

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_length = tf.constant(shape=(self.config.batch_size,), value=self.sequence_lengths, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        self.transition_matrix = K.eval(transition_params)

        return loss
