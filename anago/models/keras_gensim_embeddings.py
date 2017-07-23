import json
import os
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.layers import Embedding


def convert_embeddings(glove_input_file, word2vec_output_file,
                       embeddings_path='embeddings.npz',
                       vocab_path='map.json'):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """
    glove2word2vec(glove_input_file, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def word2vec_embedding_layer(embeddings_path='embeddings.npz'):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """

    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights])
    return layer


def word2vec_embedding_layer2(embeddings_path, vocab_path, word_index, embedding_dimension):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """
    word2idx, idx2word = load_vocab(vocab_path)
    weights = np.load(open(embeddings_path, 'rb'))
    embeddings_index = {w: weights[i] for w, i in word2idx.items()}

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector[:embedding_dimension]

    layer = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      )
    return layer


def embedding_layer(embeddings_path, word_index):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """
    weights = np.load(open(embeddings_path, 'rb'))
    embeddings_index = {w: weights[i - 2] for w, i in word_index.items()}

    embedding_matrix = np.zeros((len(word_index), weights.shape[1]))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    layer = Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      )
    return layer


if __name__ == '__main__':
    dir_name = os.path.join(os.path.dirname(__file__), '../../data/glove.6B')
    input_file = os.path.join(dir_name, 'glove.6B.300d.txt')
    output_file = os.path.join(dir_name, 'w2v.6B.300d.txt')
    convert_embeddings(input_file, output_file)

    word2idx, idx2word = load_vocab()
    embeddings = word2vec_embedding_layer()
