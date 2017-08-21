import numpy as np


def load_data_and_labels(filename):
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
                word, _, _, tag = line.split(' ')
                words.append(word)
                tags.append(tag)
    return np.asarray(sents), np.asarray(labels)


def load_glove_vocab(filename):
    """Loads GloVe's vocab from a file.

    Args:
        filename: path to the glove vectors
    Returns:
        a set of all words in GloVe
    """
    print('Building vocab...')
    with open(filename) as f:
        vocab = {line.strip().split()[0] for line in f}
    print('- done. {} tokens'.format(len(vocab)))
    return vocab


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
            embedding = [float(x) for x in line[1:dim+1]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    return embeddings


def batch_iter(dataset, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(dataset) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(dataset)
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = zip(*shuffled_data[start_index:end_index])
                if preprocessor:
                    yield preprocessor(X, y)
                else:
                    yield X, y

    return num_batches_per_epoch, data_generator()
