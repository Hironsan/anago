# -*- coding: utf-8 -*-
"""
Preprocessors.
"""
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

UNK = '<UNK>'
PAD = '<PAD>'


def normalize_number(text):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', text)


class StaticPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, lowercase=True, num_norm=True,
                 use_char=True, vocab_init=None):
        self._lowercase = lowercase
        self._num_norm = num_norm
        self._use_char = use_char
        self._vocab_init = vocab_init or {}
        self._word_vocab = {PAD: 0, UNK: 1}
        self._char_vocab = {PAD: 0, UNK: 1}
        self._label_vocab = {PAD: 0}

    def fit(self, X, y=None):

        for w in set(itertools.chain(*X)) | set(self._vocab_init):

            # create character dictionary
            if self._use_char:
                for c in w:
                    if c in self._char_vocab:
                        continue
                    self._char_vocab[c] = len(self._char_vocab)

            # create word dictionary
            if self._lowercase:
                w = w.lower()
            if self._num_norm:
                w = normalize_number(w)
            self._word_vocab[w] = len(self._word_vocab)

        # create label dictionary
        for t in set(itertools.chain(*y)):
            self._label_vocab[t] = len(self._label_vocab)

        return self

    def transform(self, X, y=None):
        words = []
        chars = []
        for sent in X:
            word_ids = []
            char_ids = []
            for w in sent:
                if self._use_char:
                    char_ids.append(self._get_char_ids(w))

                if self._lowercase:
                    w = w.lower()
                if self._num_norm:
                    w = normalize_number(w)
                word_id = self._word_vocab.get(w, self._word_vocab[UNK])
                word_ids.append(word_id)

            words.append(word_ids)
            chars.append(char_ids)

        if y is not None:
            y = [[self._label_vocab[t] for t in sent] for sent in y]

        if self._use_char:
            inputs = [words, chars]
        else:
            inputs = [words]

        return (inputs, y) if y is not None else inputs

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, docs):
        id2label = {i: t for t, i in self._label_vocab.items()}

        return [[id2label[t] for t in doc] for doc in docs]

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def _get_char_ids(self, word):
        return [self._char_vocab.get(c, self._char_vocab[UNK]) for c in word]

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, num_labels):
        self.num_labels = num_labels

    def transform(self, X, y=None):
        words, chars = X
        words = pad_sequences(words, padding='post')
        chars = pad_nested_sequences(chars)

        if y is not None:
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.num_labels)
        sents = [words, chars]

        return (sents, y) if y is not None else sents

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p


def pad_nested_sequences(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x
