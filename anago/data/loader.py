from __future__ import absolute_import, unicode_literals
import numpy as np
import os
from collections import Counter
from itertools import chain


TRAIN_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/train.txt')
VALID_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/valid.txt')
TEST_DATA = os.path.join(os.path.dirname(__file__), '../../data/conll2003/en/test.txt')
UNK = '<UNK>'


def word_mapping(words, word_preprocess=lambda x: x):
    indices_word = _fit_term_index(words, reserved=[UNK], preprocess=word_preprocess)
    word_indices = _invert_index(indices_word)

    return indices_word, word_indices


def char_mapping(words, word_preprocess=lambda x: x):
    indices_char = _fit_term_index(chain(*words), reserved=[UNK], preprocess=word_preprocess)
    char_indices = _invert_index(indices_char)

    return indices_char, char_indices


def tag_mapping(tags):
    indices_tag = _fit_term_index(tags)
    tag_indices = _invert_index(indices_tag)

    return indices_tag, tag_indices


def prepare_sentence(str_words, word_indices, char_indices, lower=False):
    """
    Prepare a sentence.
    """

    words = [convert_words_str(words, word_indices, lower) for words in str_words]
    chars = [convert_char_str(words, char_indices, lower) for words in str_words]

    return words, chars


def convert_words_str(str_words, word_indices, lower=False):
    def f(x): return x.lower() if lower else x
    words = [word_indices.get(f(w), word_indices[UNK]) for w in str_words]

    return words


def convert_char_str(str_words, char_indices, lower=False):
    def f(x): return x.lower() if lower else x
    # Skip characters that are not in the training set
    chars = [[char_indices[c] for c in f(w) if c in char_indices] for w in str_words]

    return chars


def convert_tag_str(tags, tag_indices):
    tags = [[tag_indices[t] for t in tt] for tt in tags]

    return tags


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def pad_word_chars(words, max_word_len):
    words_for = []
    for word in words:
        padding = [0] * (max_word_len - len(word))
        padded_word = word + padding
        words_for.append(padded_word[:max_word_len])
    return words_for


def pad_words(words, max_word_len, max_sent_len):
    padding = [[0] * max_word_len for i in range(max_sent_len - len(words))]
    words += padding
    return words[:max_sent_len]


def pad_chars(dataset, max_word_len, max_sent_len):
    result = []
    for sent in dataset:
        words = pad_word_chars(sent, max_word_len)
        words = pad_words(words, max_word_len, max_sent_len)
        result.append(words)
    return np.asarray(result)


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

    return words, ne_tags
