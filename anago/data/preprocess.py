import collections
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

UNK = '<UNK>'
PAD = '<PAD>'
Vocab = collections.namedtuple('Vocab', ['word', 'char', 'tag'])


def load_word_embeddings(vocab, glove_filename, dim):
    """Loads GloVe vectors in numpy array

    Arguments:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    return embeddings


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
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length
