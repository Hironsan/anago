# -*- coding: utf-8 -*-
"""
Preprocessors.
"""
import re

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from anago.utils import Vocabulary

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'


def normalize_number(text):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', text)


class IndexTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.

    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """

    def __init__(self, lower=True, num_norm=True,
                 use_char=True, initial_vocab=None):
        """Create a preprocessor object.

        Args:
            lower: boolean. Whether to convert the texts to lowercase.
            use_char: boolean. Whether to use char feature.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        """
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

    def fit(self, X, y):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        self._word_vocab.add_documents(X)
        self._label_vocab.add_documents(y)
        if self._use_char:
            for doc in X:
                self._char_vocab.add_documents(doc)

        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab.build()

        return self

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        word_ids = pad_sequences(word_ids, padding='post')

        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
            char_ids = pad_nested_sequences(char_ids)
            features = [word_ids, char_ids]
        else:
            features = word_ids

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.label_size).astype(int)
            # In 2018/06/01, to_categorical is a bit strange.
            # >>> to_categorical([[1,3]], num_classes=4).shape
            # (1, 2, 4)
            # >>> to_categorical([[1]], num_classes=4).shape
            # (1, 4)
            # So, I expand dimensions when len(y.shape) == 2.
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        """Learn vocabulary and return document id matrix.

        This is equivalent to fit followed by transform.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.

        Returns:
            list : document id matrix.
            list: label id matrix.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None):
        """Return label strings.

        Args:
            y: label id matrix.
            lengths: sentences length.

        Returns:
            list: list of list of strings.
        """
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]

        return inverse_y

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

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


class ELMoTransformer(IndexTransformer):

    def __init__(self, lower=True, num_norm=True,
                 use_char=True, initial_vocab=None):
        super(ELMoTransformer, self).__init__(lower, num_norm, use_char, initial_vocab)
        self._elmo = Elmo(options_file, weight_file, 2, dropout=0)

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """
        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        word_ids = pad_sequences(word_ids, padding='post')

        char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
        char_ids = pad_nested_sequences(char_ids)

        character_ids = batch_to_ids(X)
        elmo_embeddings = self._elmo(character_ids)['elmo_representations'][1]
        elmo_embeddings = elmo_embeddings.detach().numpy()

        features = [word_ids, char_ids, elmo_embeddings]

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self.label_size).astype(int)
            # In 2018/06/01, to_categorical is a bit strange.
            # >>> to_categorical([[1,3]], num_classes=4).shape
            # (1, 2, 4)
            # >>> to_categorical([[1]], num_classes=4).shape
            # (1, 4)
            # So, I expand dimensions when len(y.shape) == 2.
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        else:
            return features
