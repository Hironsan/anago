from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from anago.data.reader import WordPreprocessor, DataSet, load_word_embeddings
from anago.data.metrics import get_chunks
from anago.data.preprocess import pad_sequences
from anago.data.utils import Progbar, get_logger
from anago.models.keras_model import LSTMCrf


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(os.path.join(config.log_dir, 'log.txt'))

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        p = WordPreprocessor()
        p = p.fit(x_train, y_train)
        train = DataSet(x_train, y_train, preprocessor=p)
        valid = DataSet(x_valid, y_valid, preprocessor=p)
        embeddings = load_word_embeddings(p.vocab_word, self.config.glove_path, self.config.word_dim)
        self.config.char_vocab_size = len(p.vocab_char)

        model = LSTMCrf(self.config, embeddings, len(p.vocab_tag))
        model = model.build()
        model.compile(loss=self.loss,
                      optimizer=Adam(lr=self.config.learning_rate)
                      )

        for epoch in range(self.config.max_epoch):
            self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, self.config.max_epoch))

            nbatches = (len(train.sents) + self.config.batch_size - 1) // self.config.batch_size
            prog = Progbar(target=nbatches)
            for i in range(nbatches):
                words, labels = train.next_batch(self.config.batch_size)
                words, chars, labels, _ = self.pad_sequence(words, labels, len(p.vocab_tag))
                train_loss = model.train_on_batch([words, chars], labels)
                prog.update(i + 1, [('train loss', train_loss)])
            acc, f1 = self.run_evaluate(valid, model)
            self.logger.info('- dev acc {:04.2f} - f1 {:04.2f}'.format(100 * acc, 100 * f1))

    def pad_sequence(self, words, labels, ntags):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = np.asarray([to_categorical(y, num_classes=ntags) for y in labels])

        if self.config.use_char:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            word_ids, char_ids = np.asarray(word_ids), np.asarray(char_ids)
            return word_ids, char_ids, labels, sequence_lengths
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)
            word_ids = np.asarray(word_ids)
            return word_ids, labels, sequence_lengths

    def loss(self, y_true, y_pred):
        y_t = K.argmax(y_true, -1)
        y_t = tf.cast(y_t, tf.int32)
        sequence_lengths = K.argmin(y_t, -1)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_t, sequence_lengths)
        loss = tf.reduce_mean(-log_likelihood)
        self.transition_params = transition_params

        return loss

    def run_evaluate(self, test, model):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        nbatches = (len(test.sents) + self.config.batch_size - 1) // self.config.batch_size
        vocab_tag = test._preprocessor.vocab_tag
        for i in range(nbatches):
            words, labels = test.next_batch(self.config.batch_size)
            words, chars, _, sequence_lengths = self.pad_sequence(words, labels, len(vocab_tag))
            logits = model.predict_on_batch([words, chars])
            labels_pred = self.predict_batch(logits, sequence_lengths)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, vocab_tag))
                lab_pred_chunks = set(get_chunks(lab_pred, vocab_tag))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1

    def predict_batch(self, logits, sequence_lengths):
        if self.config.crf:
            viterbi_sequences = []
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, K.eval(self.transition_params))
                viterbi_sequences += [viterbi_sequence]

            return viterbi_sequences
