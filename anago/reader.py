import numpy as np


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
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
    return np.asarray(sents), np.asarray(labels)


def load_glove_vocab(filename):
    """Loads GloVe's vocab from a file.

    Args:
        filename (str): path to the glove vectors.

    Returns:
        set: a set of all words in GloVe.
    """
    print('Building vocab...')
    with open(filename) as f:
        vocab = {line.strip().split()[0] for line in f}
    print('- done. {} tokens'.format(len(vocab)))
    return vocab


def load_word_embeddings(vocab, glove_filename, dim):
    """Loads GloVe vectors in numpy array.

    Args:
        vocab (): dictionary vocab[word] = index.
        glove_filename (str): a path to a glove file.
        dim (int): dimension of embeddings.

    Returns:
        numpy array: an array of word embeddings.
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:dim+1]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    return embeddings


def load_glove(file):
    """Loads GloVe vectors in numpy array.

    Args:
        file (str): a path to a glove file.

    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file) as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model


def batch_iter(data, labels, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                if preprocessor:
                    yield preprocessor.transform(X, y)
                else:
                    yield X, y

    return num_batches_per_epoch, data_generator()