from __future__ import absolute_import, unicode_literals
import numpy as np
import os
from collections import Counter
from itertools import chain

from sklearn.model_selection import train_test_split


DATA = os.path.join(os.path.dirname(__file__), '../../data/ja/annotation.conll')
VOCAB = os.path.join(os.path.dirname(__file__), '../../data/word2vec.ja/map.jp.json')

def load_data(word_preprocess=lambda x: x):
    """Loads the conll2003 text chunking dataset.
    :param word_preprocess: The filtering function for word forms.
           for example, use lambda w: w.lower() when all words should be lowercased.
    :return: words, pos_tags, chunk_tags, ne_tags
    """
    X, y = load_file(DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    indices_word = _fit_term_index(X_train, reserved=['<PAD>', '<UNK>'], preprocess=word_preprocess)
    word_indices = _invert_index(indices_word)

    #indices_char = _fit_term_index(chain(*X_words_train), reserved=['<UNK>'], preprocess=word_preprocess)
    #char_indices = _invert_index(indices_char)

    indices_chunk = _fit_term_index(y_train, reserved=['<PAD>'])
    chunk_indices = _invert_index(indices_chunk)

    X_words_train = np.array([[word_indices.get(word_preprocess(w), word_indices['<UNK>']) for w in words] for words in X_train])
    X_words_test = np.array([[word_indices.get(word_preprocess(w), word_indices['<UNK>']) for w in words] for words in X_test])
    #X_chars_train = np.array([[[char_indices[ch] for ch in word_preprocess(word)] for word in words] for words in X_words_train])
    #X_chars_test = np.array([[[char_indices.get(ch, char_indices['<UNK>']) for ch in word_preprocess(word)] for word in words] for words in X_words_test])
    y_train = np.array([[chunk_indices[t] for t in chunk_tags] for chunk_tags in y_train])
    y_test = np.array([[chunk_indices[t] for t in chunk_tags] for chunk_tags in y_test])
    return X_words_train, y_train, X_words_test, y_test, indices_word, indices_chunk


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
        import re
        reg = re.compile(r"[0-9０１２３４５６７８９]")
        sents = []
        for i in range(len(split_indices) - 1):
            l, r = split_indices[i] + 1, split_indices[i + 1]
            sent = [reg.sub('0', line).split('\t') for line in lines[l: r]]
            if sent:
                sents.append(sent)
        return sents

    def separate_elements(sents):
        words, ne_tags = zip(*[zip(*sent) for sent in sents])
        return words, ne_tags

    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        split_indices = [i for i, line in enumerate(lines) if line == '']
        sents = separate_sentences(lines, split_indices)
        words, ne_tags = separate_elements(sents)

    return words, ne_tags
