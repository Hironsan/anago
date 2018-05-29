"""
Utility functions.
"""
import io
import os
import zipfile
from collections import Counter

import numpy as np
import requests


def download(url, save_dir='.'):
    """Download a trained weights, config and preprocessor.

    Args:
        url (str): target url.
        save_dir (str): store directory.
    """
    print('Downloading...')
    r = requests.get(url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as f:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        f.extractall(save_dir)
    print('Complete!')


def load_data_and_labels(filename):
    """Loads data and label from a file.

    Args:
        filename (str): path to the file.

        The file format is tab-separated values.
        A blank line is required at the end of a sentence.

        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O

        Peter	B-PER
        Blackburn	I-PER
        ...
        ```

    Returns:
        tuple(numpy array, numpy array): data and labels.

    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    words, tags = [], []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []

    return sents, labels


def batch_iter(data, labels, batch_size=1, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(data[0]) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data[0])
        while True:
            indices = np.arange(data_size)
            # Shuffle the data at each epoch
            if shuffle:
                indices = np.random.permutation(indices)

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = [[d[i] for i in indices[start_index: end_index]] for d in data]
                y = [labels[i] for i in indices[start_index: end_index]]
                yield preprocessor.transform(X, y)

    return num_batches_per_epoch, data_generator()


class Vocabulary(object):
    """
    A vocabulary that maps words to ints (storing a vocabulary)
    """

    def __init__(self, num_words=None, lower=True, start=1, oov_token=None):
        self._token2id = {}
        self._id2token = {}
        self._start = start
        self._lower = lower
        self._num_words = num_words
        self._token_count = Counter()

    def add_word(self, token):
        """Add token to vocabulary.

        Args:
            token (str): token to add
        """
        self._token_count.update([token])

    def add_documents(self, docs):
        for sent in docs:
            self._token_count.update(sent)

    def doc2id(self, doc):
        return [self._token2id.get(token) for token in doc]

    def build(self):
        token_freq = self._token_count.most_common(self._num_words)
        idx = self._start
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token[idx] = token
            idx += 1

    def word_id(self, word):
        """Get the word_id of given word.

        Args:
            word (str): word from vocabulary
        Returns:
            int: int id of word
        """
        return self._token2id.get(word, None)

    def __len__(self):
        return len(self._token2id)

    def id_to_word(self, wid):
        """Word-id to word (string).

        Args:
            wid (int): word id
        Returns:
            str: string of given word id
        """
        return self._id2token.get(wid)

    @property
    def vocab(self):
        """
        dict: get the dict object of the vocabulary
        """
        return self._token2id

    def reverse_vocab(self):
        """
        Return the vocabulary as a reversed dict object
        Returns:
            dict: reversed vocabulary object
        """
        return self._id2token
