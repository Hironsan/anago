from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from anago.models.model import LstmCrfModel
from anago.data.preprocess import load_word_embeddings
from anago.data.conll import extract_data, load_glove_vocab, WordPreprocessor, DataSet
from anago.data_utils import pad_sequences

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

    def train(self, x_train, y_train):
        vocab_glove = load_glove_vocab(config.glove_path)
        p = WordPreprocessor(vocab_init=vocab_glove)
        p = p.fit(x_train, y_train)
        vocab_word = p.vocab_word
        vocab_char = p.vocab_char
        vocab_tag  = p.vocab_tag
        dataset = DataSet(x_train, y_train, preprocessor=p)

        with tf.Session() as sess:
            embeddings = load_word_embeddings(vocab_word, config.glove_path, config.word_dim)
            model = LstmCrfModel(config, embeddings, vocab_char, vocab_tag)
            model.build()
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            train_op = optimizer.minimize(model.loss)
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.config.max_epoch):
                self.run_epoch(sess, dataset, train_op, model)
                self.config.learning_rate *= self.config.lr_decay

    def run_epoch(self, sess, dataset, train_op, model):
        nbatches = (len(dataset.sents) + self.config.batch_size - 1) // self.config.batch_size
        for i in range(nbatches):
            words, labels = dataset.next_batch(self.config.batch_size)

            fd, _ = model.get_feed_dict(words, labels, self.config.learning_rate, self.config.dropout)

            _, train_loss = sess.run([train_op, model.loss], feed_dict=fd)

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


if __name__ == '__main__':
    train_path = os.path.join(config.data_path, 'train.txt')
    x_train, y_train = extract_data(train_path)
    t = Trainer(config)
    t.train(x_train, y_train)
