from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os
UNK = '<UNK>'
PAD = '<PAD>'


def _read_dataset(filename, preprocess):
    with open(filename, 'r') as f:
        data = f.read().replace("-DOCSTART- -X- -X- O\n\n", "").strip().split('\n\n')
        data = [sent.split('\n') for sent in data]
        sents = [[preprocess(line.split(' ')[0]) for line in sent] for sent in data]
        entities = [[line.split(' ')[-1] for line in sent] for sent in data]
        return {'X': sents, 'y': entities}


def _build_vocab(filename, preprocess):
    data = _read_dataset(filename, preprocess)

    counter = collections.Counter(itertools.chain(*data['X']))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = [PAD, UNK] + list(words)
    word_to_id = dict(zip(words, range(len(words))))

    entities = sorted(set(itertools.chain(*data['y'])))
    entities = [PAD] + entities
    entity_to_id = dict(zip(entities, range(len(entities))))

    return word_to_id, entity_to_id


def _file_to_ids(filename, word_to_id, entity_to_id, preprocess):
    data = _read_dataset(filename, preprocess)
    sents = [[word_to_id.get(word, word_to_id[UNK]) for word in sent] for sent in data['X']]
    entities = [[entity_to_id[entity] for entity in entities] for entities in data['y']]
    return {'X': sents, 'y': entities}


def conll_raw_data(data_path=None, preprocess=str.lower):
    """Load conll raw data from data directory "data_path".
    Reads NER text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The conll dataset comes from conll's webpage:
    http://www.cnts.ua.ac.be/conll2003/ner.tgz
    Args:
        data_path: string path to the directory where simple-examples.tgz has been extracted.
        preprocess: preprocessing function for word
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, 'train.txt')
    valid_path = os.path.join(data_path, 'valid.txt')
    test_path = os.path.join(data_path, 'test.txt')

    word_to_id, entity_to_id = _build_vocab(train_path, preprocess)
    train_data = _file_to_ids(train_path, word_to_id, entity_to_id, preprocess)
    valid_data = _file_to_ids(valid_path, word_to_id, entity_to_id, preprocess)
    test_data = _file_to_ids(test_path, word_to_id, entity_to_id, preprocess)

    return train_data, valid_data, test_data, word_to_id, entity_to_id
