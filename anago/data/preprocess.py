import collections
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

UNK = '<UNK>'
PAD = '<PAD>'
Vocab = collections.namedtuple('Vocab', ['word', 'char', 'tag'])


class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, lowercase=True, num_norm=True, char_feature=True, vocab_init=None):
        self.lowercase = lowercase
        self.num_norm = num_norm
        self.char_feature = char_feature
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

    def transform(self, X, y):
        sents = []
        for sent in X:
            word_ids = []
            char_word_ids = []
            for w in sent:
                if self.char_feature:
                    char_ids = self._get_char_ids(w)
                    char_word_ids.append(char_ids)

                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)

            if self.char_feature:
                sents.append((char_word_ids, word_ids))
            else:
                sents.append(word_ids)

        y = [[self.vocab_tag[t] for t in sent] for sent in y]

        return sents, y

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


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Arguments:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
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
    Arguments:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        #max_length = max(map(lambda x: len(x), sequences))
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
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


def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """Convert class labels from scalars to one-hot vectors."""
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes))
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))


def prepare_preprocessor(X, y, use_char=True):
    p = WordPreprocessor()
    p.fit(X, y)

    def pad_sequence(words, labels, ntags):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = dense_to_one_hot(labels, ntags, nlevels=2)

        if use_char:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            word_ids, char_ids = np.asarray(word_ids), np.asarray(char_ids)
            return [word_ids, char_ids], labels
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)
            word_ids = np.asarray(word_ids)
            return word_ids, labels

    def func(X, y):
        X, y = p.transform(X, y)
        X, y = pad_sequence(X, y, len(p.vocab_tag))
        return X, y

    func.vocab_word = p.vocab_word
    func.vocab_char = p.vocab_char
    func.vocab_tag  = p.vocab_tag
    func.inverse_transform = p.inverse_transform
    return func
