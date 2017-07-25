from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os

import numpy as np
UNK = '<UNK>'
PAD = '<PAD>'


def _read_dataset(filename, processing_word, processing_tag=lambda x: x):
    with open(filename, 'r') as f:
        data = f.read().replace("-DOCSTART- -X- -X- O\n\n", "").strip().split('\n\n')
        data = [sent.split('\n') for sent in data]
        sents = [[processing_word(line.split(' ')[0]) for line in sent] for sent in data]
        entities = [[processing_tag(line.split(' ')[-1]) for line in sent] for sent in data]
        return {'X': sents, 'y': entities}


def _build_vocab(filename, preprocess):
    data = _read_dataset(filename, preprocess)

    counter = collections.Counter(itertools.chain(*data['X']))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = [PAD, UNK] + list(words)
    word_to_id = dict(zip(words, range(len(words))))

    chars = sorted(get_char_vocab(data['X']))
    chars = [PAD, UNK] + chars
    char_to_id = dict(zip(chars, range(len(chars))))

    entities = sorted(set(itertools.chain(*data['y'])))
    entities = [PAD] + entities
    entity_to_id = dict(zip(entities, range(len(entities))))

    return word_to_id, char_to_id, entity_to_id


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    for words in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    if not filename:
        return vocab
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def add_vocab(word_to_id, vocab_glove):
    i = max(word_to_id.values()) + 1
    for w in vocab_glove:
        if w not in word_to_id:
            word_to_id[w] = i
            i += 1
    return word_to_id


def load_vocab(data_path, glove_path=None, preprocess=str.lower):
    train_path = os.path.join(data_path, 'train.txt')
    word_to_id, char_to_id, entity_to_id = _build_vocab(train_path, preprocess)

    # build vocab
    vocab_glove = get_glove_vocab(glove_path)
    word_to_id = add_vocab(word_to_id, vocab_glove)

    return word_to_id, char_to_id, entity_to_id


def get_glove_vectors(vocab, glove_filename, dim):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    #limit = np.sqrt(3 / dim)
    #embeddings = np.random.uniform(-limit, limit, size=(len(vocab), dim))
    if not glove_filename:
        return embeddings
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    return embeddings


def _file_to_ids(filename, processing_word, processing_tag):
    data = _read_dataset(filename, processing_word, processing_tag)
    return data


def conll_raw_data(data_path, processing_word, processing_tag):
    """Load conll raw data from data directory "data_path".
    Reads NER text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The conll dataset comes from conll's webpage:
    http://www.cnts.ua.ac.be/conll2003/ner.tgz
    Args:
        data_path: string path to the directory where simple-examples.tgz has been extracted.
        preprocess: preprocessing function for word
        vocab_path: pre-trained vocabulary
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, 'train.txt')
    valid_path = os.path.join(data_path, 'valid.txt')
    test_path = os.path.join(data_path, 'test.txt')

    train_data = _file_to_ids(train_path, processing_word, processing_tag)
    valid_data = _file_to_ids(valid_path, processing_word, processing_tag)
    test_data = _file_to_ids(test_path, processing_word, processing_tag)

    return train_data, valid_data, test_data
