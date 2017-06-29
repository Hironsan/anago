from __future__ import absolute_import, unicode_literals
import numpy as np
import os
from collections import Counter
from itertools import chain


TRAIN_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/train.txt')
VALID_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/valid.txt')
TEST_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/test.txt')


def load_data(word_preprocess=lambda x: x):
    """Loads the conll2003 text chunking dataset.
    :param word_preprocess: The filtering function for word forms.
           for example, use lambda w: w.lower() when all words should be lowercased.
    :return: words, pos_tags, chunk_tags, ne_tags
    """
    X_words_train, _, _, y_train = load_file(TRAIN_DATA)
    X_words_valid, _, _, y_valid = load_file(VALID_DATA)
    X_words_test, _, _, y_test = load_file(TEST_DATA)

    index2word = _fit_term_index(X_words_train, reserved=['<PAD>', '<UNK>'], preprocess=word_preprocess)
    word2index = _invert_index(index2word)

    index2chunk = _fit_term_index(y_train, reserved=['<PAD>'])
    chunk2index = _invert_index(index2chunk)

    X_words_train = np.array([[word2index[word_preprocess(w)] for w in words] for words in X_words_train])
    X_words_test = np.array([[word2index.get(word_preprocess(w), word2index['<UNK>']) for w in words] for words in X_words_test])
    y_train = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in y_train])
    y_test = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in y_test])
    return X_words_train, y_train, X_words_test, y_test, index2word, index2chunk


def _fit_term_index(terms, reserved=(), preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = list(reserved) + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}


def load_file(filename):
    """Loads a conll2003 data file.
    :param filename: The requested filename.
    :return: words, pos_tags, chunk_tags, ne_tags
    """
    def separate_sentences(lines, split_indices):
        sents = []
        for i in range(len(split_indices) - 1):
            l, r = split_indices[i] + 1, split_indices[i + 1]
            sent = [line.split() for line in lines[l: r]]
            if sent:
                sents.append(sent)
        return sents

    def separate_elements(sents):
        words, pos_tags, chunk_tags, ne_tags = zip(*[zip(*sent) for sent in sents])
        return words, pos_tags, chunk_tags, ne_tags

    with open(filename) as f:
        lines = [line.strip() for line in f if not line.startswith('-DOCSTART-')]
        split_indices = [i for i, line in enumerate(lines) if line == '']
        sents = separate_sentences(lines, split_indices)
        words, pos_tags, chunk_tags, ne_tags = separate_elements(sents)

    return words, pos_tags, chunk_tags, ne_tags
