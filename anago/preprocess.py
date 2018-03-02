# -*- coding: utf-8 -*-
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
# from keras.preprocessing.sequence import pad_sequences

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
            y = [[self.label_dic[t] for t in sent] for sent in y]

        inputs = [words, chars] if self._char_feature else [words]

        return (inputs, y) if y is not None else inputs

    def inverse_transform(self, docs):
        id2label = {i: t for t, i in self.label_dic.items()}

        return [[id2label[t] for t in doc] for doc in docs]

    def _get_char_ids(self, word):
        return [self.char_dic.get(c, self.char_dic[UNK]) for c in word]


class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        # padding, sequence lengths and one-hot
        pass

    def pad_word(self):
        pass

    def pad_char(self):
        pass


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
            sents, y = self.pad_sequence(words, chars, y)
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

    def pad_sequence(self, word_ids, char_ids, labels=None):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = to_categorical(labels, len(self.vocab_tag))

        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        word_ids = np.asarray(word_ids)

        if self.char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            char_ids = np.asarray(char_ids)
            return [word_ids, char_ids], labels
        else:
            return word_ids, labels

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max(len(max(seq, key=len)) for seq in sequences)
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length


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
