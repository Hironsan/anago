from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from anago.models.model import LstmCrfModel
from anago.data.preprocess import load_word_embeddings
from anago.data.conll import load_glove_vocab, WordPreprocessor, DataSet
from anago.data_utils import pad_sequences, get_chunks
from anago.general_utils import Progbar, get_logger


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(os.path.join(config.log_dir, 'log.txt'))

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        vocab_glove = load_glove_vocab(self.config.glove_path)
        p = WordPreprocessor()
        p = p.fit(x_train, y_train)
        train = DataSet(x_train, y_train, preprocessor=p)
        valid = DataSet(x_valid, y_valid, preprocessor=p)
        embeddings = load_word_embeddings(p.vocab_word, self.config.glove_path, self.config.word_dim)
        best_score = 0
        # for early stopping
        nepoch_no_imprv = 0

        with tf.Session() as sess:
            model = LstmCrfModel(self.config, embeddings, p.vocab_char, p.vocab_tag)
            model.build()
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            train_op = optimizer.minimize(model.loss)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for epoch in range(self.config.max_epoch):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.max_epoch))

                nbatches = (len(train.sents) + self.config.batch_size - 1) // self.config.batch_size
                prog = Progbar(target=nbatches)
                for i in range(nbatches):
                    words, labels = train.next_batch(self.config.batch_size)
                    fd, _ = model.get_feed_dict(words, labels, self.config.dropout)
                    _, train_loss = sess.run([train_op, model.loss], feed_dict=fd)
                    prog.update(i + 1, [("train loss", train_loss)])

                acc, f1 = self.run_evaluate(sess, valid, model)
                self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.save_path):
                        os.makedirs(self.config.save_path)
                    saver.save(sess, self.config.save_path)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                        break

    def run_evaluate(self, sess, test, model):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
        Returns:
            accuracy
            f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        nbatches = (len(test.sents) + self.config.batch_size - 1) // self.config.batch_size
        vocab_tag = test._preprocessor.vocab_tag
        for i in range(nbatches):
            words, labels = test.next_batch(self.config.batch_size)
            labels_pred, sequence_lengths = model.predict_batch(sess, words)

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
