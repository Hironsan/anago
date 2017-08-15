import collections
import os

import numpy as np
from tensorflow.python.framework import random_seed

from anago.data.preprocess import WordPreprocessor


def extract_data(filename):
    sents, labels = [], []
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, _, _, tag = line.split(' ')
                words.append(word)
                tags.append(tag)
    return np.asarray(sents), np.asarray(labels)


class DataSet(object):

    def __init__(self, sents, labels, seed=None, preprocessor=None):
        """Construct a DataSet."""
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        assert sents.shape[0] == labels.shape[0]
        self._num_examples = sents.shape[0]
        self._sents = sents
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._preprocessor = preprocessor

    @property
    def sents(self):
        return self._sents

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._sents = self.sents[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            sents_rest_part = self._sents[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._sents = self.sents[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            sents_new_part = self._sents[start:end]
            labels_new_part = self._labels[start:end]
            X, y = np.concatenate((sents_rest_part, sents_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
            # return np.concatenate((sents_rest_part, sents_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            X, y = self._sents[start:end], self._labels[start:end]
            #return self._sents[start:end], self._labels[start:end]
        if self._preprocessor:
            return self._preprocessor.transform(X, y)
        else:
            return X, y


def read_datasets(train_dir, glove_file, one_hot=False, valid_size=5000, seed=None):
    train_path = os.path.join(train_dir, 'train.txt')
    valid_path = os.path.join(train_dir, 'valid.txt')
    test_path = os.path.join(train_dir, 'test.txt')
    x_train, y_train = extract_data(train_path)
    x_valid, y_valid = extract_data(valid_path)
    x_test, y_test = extract_data(test_path)

    vocab_glove = load_glove_vocab(glove_file)

    p = WordPreprocessor(vocab_init=vocab_glove)
    p = p.fit(np.concatenate((x_train, x_valid, x_test)), y_train)

    train = DataSet(x_train, y_train, seed=seed, preprocessor=p)
    valid = DataSet(x_valid, y_valid, seed=seed, preprocessor=p)
    test = DataSet(x_test, y_test, seed=seed, preprocessor=p)
    Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])

    return Datasets(train=train, valid=valid, test=test)


def load_glove_vocab(filename):
    """Loads GloVe's vocab from a file.

    Args:
        filename: path to the glove vectors
    Returns:
        a set of all words in GloVe
    """
    print('Building vocab...')
    with open(filename) as f:
        vocab = {line.strip().split()[0] for line in f}
    print('- done. {} tokens'.format(len(vocab)))
    return vocab