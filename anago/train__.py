from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

from anago.models.model import LstmCrfModel
from anago.data.preprocess import load_word_embeddings
from anago.data.conll import extract_data, load_glove_vocab, WordPreprocessor, DataSet
from anago.data_utils import pad_sequences, get_chunks
from anago.general_utils import Progbar, get_logger

parser = argparse.ArgumentParser()
# data settings
parser.add_argument('--data_path', required=True, help='Where the training/test data is stored.')
parser.add_argument('--save_path', required=True, help='Where the trained model is stored.')
parser.add_argument('--log_dir', help='Where log data is stored.')
parser.add_argument('--glove_path', default=None, help='Where GloVe embedding is stored.')
# model settings
parser.add_argument('--dropout', default=0.5, type=float, help='The probability of keeping weights in the dropout layer')
parser.add_argument('--char_dim', default=100, type=int, help='Character embedding dimension')
parser.add_argument('--word_dim', default=300, type=int, help='Word embedding dimension')
parser.add_argument('--lstm_size', default=300, type=int, help='The number of hidden units in lstm')
parser.add_argument('--char_lstm_size', default=100, type=int, help='The number of hidden units in char lstm')
parser.add_argument('--use_char', default=True, help='Use character feature', action='store_false')
parser.add_argument('--crf', default=True, help='Use CRF', action='store_false')
parser.add_argument('--train_embeddings', default=True, help='Fine-tune word embeddings', action='store_false')
# learning settings
parser.add_argument('--batch_size', default=20, type=int, help='The batch size')
parser.add_argument('--clip_value', default=0.0, type=float, help='The clip value')
parser.add_argument('--learning_rate', default=0.001, type=float, help='The initial value of the learning rate')
parser.add_argument('--lr_decay', default=0.9, type=float, help='The decay of the learning rate for each epoch')
parser.add_argument('--lr_method', default='adam', help='The learning method')
parser.add_argument('--max_epoch', default=15, type=int, help='The number of epochs')
parser.add_argument('--reload', default=False, help='Reload model', action='store_true')
parser.add_argument('--nepoch_no_imprv', default=3, type=int, help='For early stopping')
config = parser.parse_args()


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.word_ids = tf.placeholder(tf.int32, (None, None), name='word_ids')
        self.char_ids = tf.placeholder(tf.int32, (None, None, None), name='char_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, (None), name='sequence_lengths')
        self.word_lengths = tf.placeholder(tf.int32, (None, None), name='word_lengths')
        self.labels = tf.placeholder(tf.int32, (None, None), name='labels')
        self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.logger = get_logger(os.path.join(config.log_dir, 'log.txt'))

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        vocab_glove = load_glove_vocab(config.glove_path)
        # p = WordPreprocessor(vocab_init=vocab_glove)
        p = WordPreprocessor()
        p = p.fit(x_train, y_train)
        vocab_word = p.vocab_word
        vocab_char = p.vocab_char
        vocab_tag  = p.vocab_tag
        dataset = DataSet(x_train, y_train, preprocessor=p)
        valid = DataSet(x_valid, y_valid, preprocessor=p)

        best_score = 0
        # for early stopping
        nepoch_no_imprv = 0

        with tf.Session() as sess:
            embeddings = load_word_embeddings(vocab_word, config.glove_path, config.word_dim)
            model = LstmCrfModel(config, embeddings, vocab_char, vocab_tag)
            model.build()
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            train_op = optimizer.minimize(model.loss)
            sess.run(tf.global_variables_initializer())
            self.add_summary(sess)
            saver = tf.train.Saver()

            for epoch in range(self.config.max_epoch):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.max_epoch))

                acc, f1 = self.run_epoch(sess, dataset, valid, train_op, model, epoch)
                self.config.learning_rate *= self.config.lr_decay

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

    def run_epoch(self, sess, train, dev, train_op, model, epoch):
        nbatches = (len(train.sents) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        for i in range(nbatches):
            words, labels = train.next_batch(self.config.batch_size)

            fd, _ = model.get_feed_dict(words, labels, self.config.learning_rate, self.config.dropout)

            _, train_loss = sess.run([train_op, model.loss], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            #if i % 10 == 0:
             #   self.file_writer.add_summary(summary, epoch * nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev, model)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, f1

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

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of words.
                A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.use_char:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_char:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr:
            feed[self.lr] = lr

        if dropout:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)


if __name__ == '__main__':
    train_path = os.path.join(config.data_path, 'train.txt')
    valid_path = os.path.join(config.data_path, 'valid.txt')
    x_train, y_train = extract_data(train_path)
    x_valid, y_valid = extract_data(valid_path)
    t = Trainer(config)
    t.train(x_train, y_train, x_valid, y_valid)
