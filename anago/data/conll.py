import collections
import os

import numpy as np
from tensorflow.python.framework import random_seed


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


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self, sents, labels, seed=None):
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
            return np.concatenate((sents_rest_part, sents_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._sents[start:end], self._labels[start:end]


def read_data_sets(train_dir, one_hot=False, valid_size=5000, seed=None):
    train_path = os.path.join(train_dir, 'train.txt')
    valid_path = os.path.join(train_dir, 'valid.txt')
    test_path = os.path.join(train_dir, 'test.txt')
    x_train, y_train = extract_data(train_path)
    x_valid, y_valid = extract_data(valid_path)
    x_test, y_test = extract_data(test_path)

    train = DataSet(x_train, y_train, seed=seed)
    valid = DataSet(x_valid, y_valid, seed=seed)
    test = DataSet(x_test, y_test, seed=seed)
    Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])

    return Datasets(train=train, valid=valid, test=test)
