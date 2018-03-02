# -*- coding: utf-8 -*-
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
                 char_feature=True, vocab_init=None):
        self._lowercase = lowercase
        self._num_norm = num_norm
        self._char_feature = char_feature
        self._vocab_init = vocab_init or {}
        self.word_dic = {PAD: 0, UNK: 1}
        self.char_dic = {PAD: 0, UNK: 1}
        self.label_dic = {PAD: 0}

    def fit(self, X, y=None):

        for w in set(itertools.chain(*X)) | set(self._vocab_init):

            # create character dictionary
            if self._char_feature:
                for c in w:
                    if c in self.char_dic:
                        continue
                    self.char_dic[c] = len(self.char_dic)

            # create word dictionary
            if self._lowercase:
                w = w.lower()
            if self._num_norm:
                w = normalize_number(w)
            self.word_dic[w] = len(self.word_dic)

        # create label dictionary
        for t in set(itertools.chain(*y)):
            self.label_dic[t] = len(self.label_dic)

        return self

    def transform(self, X, y=None):
        words = []
        chars = []
        for sent in X:
            word_ids = []
            char_ids = []
            for w in sent:
                if self._char_feature:
                    char_ids.append(self._get_char_ids(w))

                if self._lowercase:
                    w = w.lower()
                if self._num_norm:
                    w = normalize_number(w)
                word_id = self.word_dic.get(w, self.word_dic[UNK])
                word_ids.append(word_id)

            words.append(word_ids)
            chars.append(char_ids)

        if y is not None:
            y = np.array([[self.label_dic[t] for t in sent] for sent in y])

        if self._char_feature:
            inputs = np.array(list(zip(words, chars)))
        else:
            inputs = words

        return (inputs, y) if y is not None else inputs

    def inverse_transform(self, docs):
        id2label = {i: t for t, i in self.label_dic.items()}

        return [[id2label[t] for t in doc] for doc in docs]

    def _get_char_ids(self, word):
        return [self.char_dic.get(c, self.char_dic[UNK]) for c in word]


class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, n_labels):
        # padding, sequence lengths and one-hot
        self.n_labels = n_labels

    def transform(self, X, y=None):
        words, chars = X
        words = pad_sequences(words, padding='post')
        chars = pad_char(chars)

        lengths = [len(y_) for y_ in y]
        y = pad_sequences(y, padding='post')
        y = to_categorical(y, self.n_labels)
        sents = [words, chars, lengths]

        return sents, y



class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.char_feature = char_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.vocab_word = None
        self.vocab_char = None
        self.vocab_tag  = None
        self.vocab_init = vocab_init or {}

    def fit(self, X, y):
        words = {PAD: 0, UNK: 1}
        chars = {PAD: 0, UNK: 1}
        tags  = {PAD: 0}

        for w in set(itertools.chain(*X)) | set(self.vocab_init):
            if not self.char_feature:
                continue
            for c in w:
                if c not in chars:
                    chars[c] = len(chars)

            w = self._lower(w)
            w = self._normalize_num(w)
            if w not in words:
                words[w] = len(words)

        for t in itertools.chain(*y):
            if t not in tags:
                tags[t] = len(tags)

        self.vocab_word = words
        self.vocab_char = chars
        self.vocab_tag  = tags

        return self

    def transform(self, X, y=None):
        """transforms input(s)

        Args:
            X: list of list of words
            y: list of list of tags

        Returns:
            numpy array: sentences
            numpy array: tags

        Examples:
            >>> X = [['President', 'Obama', 'is', 'speaking']]
            >>> print(self.transform(X))
            [
                [
                    [1999, 1037, 22123, 48388],       # word ids
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # list of char ids
                        [1, 2, 3, 4, 5, 0, 0, 0, 0],  # 0 is a pad
                        [1, 2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 6, 7, 8, 0]
                    ]
                ]
            ]
        """
        words = []
        chars = []
        lengths = []
        for sent in X:
            word_ids = []
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                if self.char_feature:
                    char_ids.append(self._get_char_ids(w))

                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)

            words.append(word_ids)
            if self.char_feature:
                chars.append(char_ids)

        if y is not None:
            y = [[self.vocab_tag[t] for t in sent] for sent in y]

        if self.padding:
            words = pad_sequences(words, padding='post')
            chars = pad_char(chars)
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, len(self.vocab_tag))
            sents = [words, chars]
        else:
            sents = [words, chars]

        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def _get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def _lower(self, word):
        return word.lower() if self.lowercase else word

    def _normalize_num(self, word):
        if self.num_norm:
            return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
        else:
            return word

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def pad_char(sequences):
    maxlen_word = max(len(max(seq, key=len)) for seq in sequences)
    maxlen_seq = len(max(sequences, key=len))
    sequences = [seq + [[] for i in range(max(maxlen_seq - len(seq), 0))] for seq in sequences]

    return np.array([pad_sequences(seq, padding='post', maxlen=maxlen_word) for seq in sequences])


def filter_embeddings(embeddings, vocab, dim):
    """Loads GloVe vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word]

    return _embeddings
